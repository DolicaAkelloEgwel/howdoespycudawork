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
import numpy as np

if DTYPE == "float32":
    C_DTYPE = "float"
else:
    C_DTYPE = "double"

LIB_NAME = "pycuda"

# Initialise PyCuda Driver
drv.init()
drv.Device(0).make_context()


def synchronise():
    drv.Context.synchronize()


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


def _send_arrays_to_gpu(cpu_arrays, n_gpu_arrs_needed):

    gpu_arrays = []

    for cpu_array in cpu_arrays:
        try:
            gpu_arrays.append(gpuarray.to_gpu(cpu_array))
        except drv.MemoryError as e:
            free_memory_pool(gpu_arrays)
            print_memory_info_after_transfer_failure(cpu_array, n_gpu_arrs_needed)
            print(e)
            return []

    return gpu_arrays


def timed_send_arrays_to_gpu(cpu_arrays):
    start = get_time()
    gpu_arrays = _send_arrays_to_gpu(cpu_arrays)
    end = get_time()
    cpu_to_gpu_time = end - start
    return cpu_to_gpu_time, gpu_arrays


class PyCudaImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.lib_name = LIB_NAME

    def timed_imaging_operation(
        self, runs, alg, alg_name, n_arrs_needed, n_gpu_arrs_needed
    ):

        n_partitions_needed = num_partitions_needed(
            self.cpu_arrays[0], n_gpu_arrs_needed, get_free_bytes()
        )

        transfer_time = 0
        operation_time = 0

        if n_partitions_needed == 1:

            cpu_result_array = np.empty_like(self.cpu_arrays[0])

            # Time transfer from CPU to GPU
            start = get_time()
            gpu_input_arrays = _send_arrays_to_gpu(
                self.cpu_arrays[:n_arrs_needed], n_gpu_arrs_needed
            )
            gpu_output_array = _send_arrays_to_gpu(
                [cpu_result_array], n_gpu_arrs_needed
            )[0]
            transfer_time += get_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += time_function(
                    lambda: alg(*gpu_input_arrays[:n_arrs_needed], gpu_output_array)
                )

            # Time the transfer from GPU to CPU
            transfer_time += time_function(lambda: gpu_output_array.get)

            # Free the GPU arrays
            free_memory_pool(gpu_input_arrays + [gpu_output_array])

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

                cpu_result_array = np.empty_like(split_cpu_arrays[i])

                # Time transferring the segments to the GPU
                start = get_time()
                gpu_input_arrays = _send_arrays_to_gpu(
                    split_cpu_arrays, n_gpu_arrs_needed
                )
                gpu_output_array_list = _send_arrays_to_gpu(
                    [cpu_result_array], n_gpu_arrs_needed
                )
                transfer_time += get_time() - start

                if not gpu_input_arrays:
                    return 0

                if not gpu_output_array_list:
                    return 0

                gpu_output_array = gpu_output_array_list[0]

                # Carry out the operation on the slices
                for _ in range(runs):
                    operation_time += time_function(
                        lambda: alg(*gpu_input_arrays[:n_arrs_needed], gpu_output_array)
                    )

                transfer_time += time_function(lambda: gpu_output_array.get)

                # Free the GPU arrays
                free_memory_pool(gpu_input_arrays + [gpu_output_array])

        if transfer_time > 0 and operation_time > 0:
            self.print_operation_times(operation_time, alg_name, runs, transfer_time)

        synchronise()

        return transfer_time + operation_time / runs
