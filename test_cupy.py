import time

import cupy as cp
import numpy as np

from imagingtester import (
    ImagingTester,
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
    ARRAY_SIZES,
    create_arrays,
    N_RUNS,
    SIZES_SUBSET,
    DTYPE,
    NO_PRINT,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
)
from write_and_read_results import write_results_to_file

LIB_NAME = "cupy"


class CupyImplementation(ImagingTester):
    def __init__(self, size, dtype, pinned_memory=False):
        super().__init__(size, dtype)

        if pinned_memory:
            self._send_arrays_to_gpu = self._send_arrays_to_gpu_with_pinned_memory
        else:
            self._send_arrays_to_gpu = self._send_arrays_to_gpu_without_pinned_memory

        self.warm_up()
        self.lib_name = LIB_NAME

    def warm_up(self):
        """
        Give CUDA an opportunity to compile these functions?
        """
        warm_up_arrays = [
            cp.asarray(cpu_array) for cpu_array in create_arrays((1, 1, 1), self.dtype)
        ]
        self.background_correction(
            *warm_up_arrays, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
        )
        self.add_arrays(*warm_up_arrays[:2])

    def free_memory_pool(self):
        """
        Delete the existing GPU arrays so that successive calls to `_send_arrays_to_gpu` don't cause any problems.
        """
        del self.gpu_arrays
        mempool.free_all_blocks()

    @staticmethod
    def _create_pinned_memory(cpu_array):
        """
        Use pinned memory as opposed to `asarray`. This allegedly this makes transferring quicker.
        :param cpu_array: The numpy array.
        :return:
        """
        mem = cp.cuda.alloc_pinned_memory(cpu_array.nbytes)
        src = np.frombuffer(mem, cpu_array.dtype, cpu_array.size).reshape(
            cpu_array.shape
        )
        src[...] = cpu_array
        return src

    def _send_arrays_to_gpu_with_pinned_memory(self):
        """
        Transfer the arrays to the GPU using pinned memory.
        """
        self.gpu_arrays = []

        for i in range(len(self.cpu_arrays)):
            pinned_memory = self._create_pinned_memory(self.cpu_arrays[i])
            array_stream = cp.cuda.Stream(non_blocking=True)
            gpu_array = cp.empty(pinned_memory.shape, dtype=self.dtype)
            gpu_array.set(pinned_memory, stream=array_stream)
            self.gpu_arrays.append(gpu_array)

    def _send_arrays_to_gpu_without_pinned_memory(self):
        """
        Transfer the arrays to the GPU without using pinned memory.
        """
        self.gpu_arrays = [cp.asarray(cpu_array) for cpu_array in self.cpu_arrays]

    @staticmethod
    def time_function(func):
        """
        Time an operation using a call to cupy's deviceSynchronize.
        :param func: The function to be timed.
        :return: The time the function took to complete its execution in seconds.
        """
        start = time.time()
        func()
        cp.cuda.runtime.deviceSynchronize()
        end = time.time()
        return end - start

    @staticmethod
    def add_arrays(arr1, arr2):
        """
        Add two arrays. Guaranteed to be slower on the GPU as it's a simple operation.
        :param arr1:
        :param arr2:
        """
        cp.add(arr1, arr2)

    def timed_add_arrays(self, runs):
        operation_time = 0

        transfer_time = self.time_function(self._send_arrays_to_gpu)

        for _ in range(runs):
            operation_time += self.time_function(
                lambda: self.add_arrays(*self.gpu_arrays[:2])
            )

        transfer_time += self.time_function(self.gpu_arrays[0].get)
        self.print_operation_times(operation_time, "adding", runs, transfer_time)

        return transfer_time + operation_time / runs

    @staticmethod
    def background_correction(data, dark, flat, clip_min, clip_max):
        norm_divide = np.subtract(flat, dark)
        norm_divide[norm_divide == 0] = clip_min
        cp.subtract(data, dark, out=data)
        cp.true_divide(data, norm_divide, out=data)
        cp.clip(data, clip_min, clip_max, out=data)

    def timed_background_correction(self, runs):
        operation_time = 0

        transfer_time = self.time_function(self._send_arrays_to_gpu)
        data, dark, flat = self.gpu_arrays

        for _ in range(runs):
            operation_time += self.time_function(
                lambda: self.background_correction(
                    data, dark, flat, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
                )
            )

        transfer_time += self.time_function(data.get)
        self.print_operation_times(
            operation_time, "background correction", runs, transfer_time
        )

        return transfer_time + operation_time / runs


# Use the maximum GPU memory
mempool = cp.get_default_memory_pool()
with cp.cuda.Device(0):
    mempool.set_limit(fraction=1)


def print_memory_metrics():
    """
    Print some information about how much space is being used on the GPU.
    """
    if NO_PRINT:
        return
    print("Used bytes:", mempool.used_bytes(), "/ Total bytes:", mempool.total_bytes())


for use_pinned_memory in [True, False]:

    # Create empty lists for storing results
    add_arrays = []
    background_correction = []

    for size in ARRAY_SIZES[:SIZES_SUBSET]:
        try:

            imaging_obj = CupyImplementation(size, DTYPE, use_pinned_memory)

            avg_add = imaging_obj.timed_add_arrays(N_RUNS)
            imaging_obj.free_memory_pool()

            avg_bc = imaging_obj.timed_background_correction(N_RUNS)
            imaging_obj.free_memory_pool()
            del imaging_obj

        except cp.cuda.memory.OutOfMemoryError as e:
            print(e)
            print("Unable to make GPU arrays with size", size)
            break

        add_arrays.append(avg_add)
        background_correction.append(avg_bc)

    if use_pinned_memory:
        memory_string = "with pinned memory"
    else:
        memory_string = "without pinned memory"

    write_results_to_file([LIB_NAME, ADD_ARRAYS, memory_string], add_arrays)
    write_results_to_file(
        [LIB_NAME, BACKGROUND_CORRECTION, memory_string], background_correction
    )
