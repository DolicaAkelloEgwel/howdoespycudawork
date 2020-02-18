import time

import cupy as cp
import numpy as np

from imagingtester import (
    ImagingTester,
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
    create_arrays,
    N_RUNS,
    SIZES_SUBSET,
    DTYPE,
    NO_PRINT,
    partition_arrays,
    USE_NONPINNED_MEMORY,
)
from imagingtester import num_partitions_needed as number_of_partitions_needed
from write_and_read_results import (
    write_results_to_file,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
    ARRAY_SIZES,
)

pinned_memory_mode = [True, False]

if USE_NONPINNED_MEMORY:
    pinned_memory_mode = pinned_memory_mode[:1]

LIB_NAME = "cupy"


def num_partitions_needed(cpu_arrays):
    return number_of_partitions_needed(cpu_arrays, mempool.get_limit())


def get_synchronized_time():
    cp.cuda.runtime.deviceSynchronize()
    return time.time()


def free_memory_pool(arrays):
    """
    Delete the existing GPU arrays so that successive calls to `_send_arrays_to_gpu` don't cause any problems.
    """
    for arr in arrays:
        del arr
        arr = None
    mempool.free_all_blocks()


def _create_pinned_memory(cpu_array):
    """
    Use pinned memory as opposed to `asarray`. This allegedly this makes transferring quicker.
    :param cpu_array: The numpy array.
    :return:
    """
    mem = cp.cuda.alloc_pinned_memory(cpu_array.nbytes)
    src = np.frombuffer(mem, cpu_array.dtype, cpu_array.size).reshape(cpu_array.shape)
    src[...] = cpu_array
    return src


def time_function(func):
    """
    Time an operation using a call to cupy's deviceSynchronize.
    :param func: The function to be timed.
    :return: The time the function took to complete its execution in seconds.
    """
    start = get_synchronized_time()
    func()
    end = get_synchronized_time()
    return end - start


def _send_arrays_to_gpu_without_pinned_memory(cpu_arrays):
    """
    Transfer the arrays to the GPU without using pinned memory.
    """
    gpu_arrays = [cp.asarray(cpu_array) for cpu_array in cpu_arrays]
    return gpu_arrays


def add_arrays(arr1, arr2):
    """
    Add two arrays. Guaranteed to be slower on the GPU as it's a simple operation.
    """
    cp.add(arr1, arr2)


def background_correction(data, dark, flat, clip_min, clip_max):
    norm_divide = np.subtract(flat, dark)
    norm_divide[norm_divide == 0] = clip_min
    cp.subtract(data, dark, out=data)
    cp.true_divide(data, norm_divide, out=data)
    cp.clip(data, clip_min, clip_max, out=data)


class CupyImplementation(ImagingTester):
    def __init__(self, size, dtype, pinned_memory=False):
        super().__init__(size, dtype)

        if pinned_memory:
            self._send_arrays_to_gpu = self._send_arrays_to_gpu_with_pinned_memory
        else:
            self._send_arrays_to_gpu = _send_arrays_to_gpu_without_pinned_memory

        self.warm_up()
        self.lib_name = LIB_NAME

    def warm_up(self):
        """
        Give CUDA an opportunity to compile these functions?
        """
        warm_up_arrays = [
            cp.asarray(cpu_array) for cpu_array in create_arrays((1, 1, 1), self.dtype)
        ]
        background_correction(*warm_up_arrays, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE)
        add_arrays(*warm_up_arrays[:2])

    def _send_arrays_to_gpu_with_pinned_memory(self, cpu_arrays):
        """
        Transfer the arrays to the GPU using pinned memory.
        """
        gpu_arrays = []

        for i in range(len(cpu_arrays)):
            pinned_memory = _create_pinned_memory(cpu_arrays[i])
            array_stream = cp.cuda.Stream(non_blocking=True)
            gpu_array = cp.empty(pinned_memory.shape, dtype=self.dtype)
            gpu_array.set(pinned_memory, stream=array_stream)
            gpu_arrays.append(gpu_array)

        return gpu_arrays

    def timed_add_arrays(self, runs):

        n_partitions_needed = num_partitions_needed(self.cpu_arrays)

        transfer_time = 0
        operation_time = 0

        if n_partitions_needed == 1:

            start = get_synchronized_time()
            gpu_arrays = self._send_arrays_to_gpu(self.cpu_arrays[:2])
            end = get_synchronized_time()
            transfer_time = end - start

            for _ in range(runs):
                operation_time += time_function(lambda: add_arrays(*gpu_arrays[:2]))

            transfer_time += time_function(gpu_arrays[0].get)

            free_memory_pool(gpu_arrays)

        else:

            split_arrays = partition_arrays(self.cpu_arrays[:2], n_partitions_needed)

            print(n_partitions_needed)
            print(split_arrays[0][0].shape)

            for i in range(n_partitions_needed):

                split_cpu_arrays = [split_array[i] for split_array in split_arrays]

                try:
                    start = get_synchronized_time()
                    gpu_arrays = self._send_arrays_to_gpu(split_cpu_arrays)
                    end = get_synchronized_time()
                    transfer_time += end - start
                except cp.cuda.memory.OutOfMemoryError:
                    print(
                        "Failed to make two GPU arrays of size",
                        split_cpu_arrays[0][0].shape,
                    )
                    print("Free bytes", mempool.free_bytes())
                    print("Limit", mempool.total_bytes())
                    print(
                        "Space needed",
                        sum([split_array.nbytes for split_array in split_cpu_arrays]),
                    )
                    break

                for _ in range(runs):
                    operation_time += time_function(lambda: add_arrays(*gpu_arrays[:2]))

                transfer_time += time_function(gpu_arrays[0].get)

                print("Completed one partition.", i)

                free_memory_pool(split_cpu_arrays + gpu_arrays)

        self.print_operation_times(operation_time, "adding", runs, transfer_time)

        return transfer_time + operation_time / runs

    def timed_background_correction(self, runs):

        n_partitions_needed = num_partitions_needed(self.cpu_arrays)

        operation_time = 0
        transfer_time = 0

        if n_partitions_needed == 1:

            start = get_synchronized_time()
            gpu_arrays = self._send_arrays_to_gpu(self.cpu_arrays)
            end = get_synchronized_time()
            transfer_time = end - start

            for _ in range(runs):
                operation_time += time_function(
                    lambda: background_correction(
                        *gpu_arrays, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
                    )
                )

            transfer_time += time_function(gpu_arrays[0].get)
            free_memory_pool(gpu_arrays)

        else:

            split_arrays = partition_arrays(self.cpu_arrays, n_partitions_needed)

            for i in range(n_partitions_needed):

                split_cpu_arrays = [split_array[i] for split_array in split_arrays]

                try:
                    start = get_synchronized_time()
                    gpu_arrays = self._send_arrays_to_gpu(split_cpu_arrays)
                    end = get_synchronized_time()
                    transfer_time += end - start
                except cp.cuda.memory.OutOfMemoryError:
                    print(
                        "Failed to make three GPU arrays of size",
                        split_cpu_arrays[0][0].shape,
                    )
                    print("Free bytes", mempool.free_bytes())
                    print("Limit", mempool.total_bytes())
                    print(
                        "Space needed",
                        sum([split_array.nbytes for split_array in split_cpu_arrays]),
                    )
                    break

                for _ in range(runs):
                    operation_time += time_function(
                        lambda: background_correction(
                            *gpu_arrays, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
                        )
                    )

                transfer_time += time_function(gpu_arrays[0].get)

                free_memory_pool(split_cpu_arrays + gpu_arrays)

        self.print_operation_times(
            operation_time, "background correction", runs, transfer_time
        )
        return transfer_time + operation_time / runs


# Use the maximum GPU memory
mempool = cp.get_default_memory_pool()
with cp.cuda.Device(0):
    mempool.set_limit(fraction=0.7)


def print_memory_metrics():
    """
    Print some information about how much space is being used on the GPU.
    """
    if NO_PRINT:
        return
    print("Used bytes:", mempool.used_bytes(), "/ Total bytes:", mempool.total_bytes())


for use_pinned_memory in pinned_memory_mode:

    # Create empty lists for storing results
    add_arrays = []
    background_correction = []

    for size in ARRAY_SIZES[:SIZES_SUBSET]:

        imaging_obj = CupyImplementation(size, DTYPE, use_pinned_memory)

        try:

            cp.cuda.runtime.deviceSynchronize()
            avg_add = imaging_obj.timed_add_arrays(N_RUNS)
            cp.cuda.runtime.deviceSynchronize()
            avg_bc = imaging_obj.timed_background_correction(N_RUNS)

            add_arrays.append(avg_add)
            background_correction.append(avg_bc)

        except cp.cuda.memory.OutOfMemoryError:
            break

    if use_pinned_memory:
        memory_string = "with pinned memory"
    else:
        memory_string = "without pinned memory"

    write_results_to_file([LIB_NAME, memory_string], ADD_ARRAYS, add_arrays)
    write_results_to_file(
        [LIB_NAME, memory_string], BACKGROUND_CORRECTION, background_correction
    )
