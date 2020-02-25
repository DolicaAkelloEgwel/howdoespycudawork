import time

from imagingtester import (
    ImagingTester,
    num_partitions_needed,
    memory_needed_for_arrays,
    get_array_partition_indices,
    DTYPE,
)
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

if DTYPE == "float32":
    C_DTYPE = "float"
else:
    C_DTYPE = "double"

LIB_NAME = "pycuda"

# Initialise PyCuda Driver
drv.init()
drv.Device(0).make_context()


def get_time():
    drv.Context.synchronize()
    return time.time()


def free_memory_pool(gpu_arrays):
    for gpu_array in gpu_arrays:
        gpu_array.gpudata.free()


def time_function(func):
    start = get_time()
    func()
    end = get_time()
    return end - start


def timed_get_array_from_gpu(gpu_array):
    start = get_time()
    gpu_array.get()
    end = get_time()
    return end - start


def get_free_bytes():
    return drv.mem_get_info()[0]


def get_total_bytes():
    return drv.mem_get_info()[1]


def get_used_bytes():
    return get_total_bytes() - get_free_bytes()


def print_memory_info_after_transfer_failure(cpu_array, n_gpu_arrs_needed):
    print(
        "Failed to make %s GPU arrays of size %s."
        % (n_gpu_arrs_needed, cpu_array.shape)
    )
    print(
        "Used bytes:",
        get_used_bytes(),
        "/ Total bytes:",
        get_total_bytes(),
        "/ Space needed:",
        memory_needed_for_arrays(cpu_array, n_gpu_arrs_needed),
    )


class PyCudaImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.lib_name = LIB_NAME
        self.streams = []

    def _send_arrays_to_gpu(self, cpu_arrays, n_gpu_arrs_needed):

        gpu_arrays = []

        for cpu_array in cpu_arrays:
            try:
                gpu_array = gpuarray.GPUArray(
                    shape=cpu_array.shape, dtype=cpu_array.dtype
                )
            except drv.MemoryError as e:
                free_memory_pool(gpu_arrays)
                print_memory_info_after_transfer_failure(cpu_array, n_gpu_arrs_needed)
                print(e)
                return []
            stream = drv.Stream()
            self.streams.append(stream)
            gpu_array.set_async(cpu_array, stream)
            gpu_arrays.append(gpu_array)

        return gpu_arrays

    def synchronise(self):
        for stream in self.streams:
            stream.synchronize()
        drv.Context.synchronize()

    def timed_imaging_operation(
        self, runs, alg, alg_name, n_arrs_needed, n_gpu_arrs_needed
    ):

        n_partitions_needed = num_partitions_needed(
            self.cpu_arrays[0], n_gpu_arrs_needed, get_free_bytes()
        )

        transfer_time = 0
        operation_time = 0

        if n_partitions_needed == 1:

            # Time transfer from CPU to GPU
            start = get_time()
            gpu_input_arrays = self._send_arrays_to_gpu(
                self.cpu_arrays[:n_arrs_needed], n_gpu_arrs_needed
            )
            transfer_time += get_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += time_function(
                    lambda: alg(*gpu_input_arrays[:n_arrs_needed])
                )

            # Time the transfer from GPU to CPU
            transfer_time += time_function(lambda: gpu_input_arrays[0].get_async)

            # Free the GPU arrays
            free_memory_pool(gpu_input_arrays)

        else:

            # Determine the number of partitions required again (to be on the safe side)
            n_partitions_needed = num_partitions_needed(
                self.cpu_arrays[0], n_gpu_arrs_needed, get_free_bytes()
            )

            indices = get_array_partition_indices(
                self.cpu_arrays[0].shape[0], n_partitions_needed
            )

            for i in range(n_partitions_needed):

                # Retrieve the segments used for this iteration of the operation
                split_cpu_arrays = [
                    cpu_array[indices[i][0] : indices[i][1] :, :]
                    for cpu_array in self.cpu_arrays
                ]

                # Time transferring the segments to the GPU
                start = get_time()
                gpu_input_arrays = self._send_arrays_to_gpu(
                    split_cpu_arrays, n_gpu_arrs_needed
                )
                transfer_time += get_time() - start

                if not gpu_input_arrays:
                    return 0

                # Carry out the operation on the slices
                for _ in range(runs):
                    operation_time += time_function(
                        lambda: alg(*gpu_input_arrays[:n_arrs_needed])
                    )

                transfer_time += time_function(lambda: gpu_input_arrays[0].get_async)

                # Free the GPU arrays
                free_memory_pool(gpu_input_arrays)

        if transfer_time > 0 and operation_time > 0:
            self.print_operation_times(operation_time, alg_name, runs, transfer_time)

        self.synchronise()

        return transfer_time + operation_time / runs
