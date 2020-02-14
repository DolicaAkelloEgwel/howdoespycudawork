import cupy as cp
import time

from cupy.cuda.stream import Event
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

from imagingtester import (
    ImagingTester,
    ARRAY_SIZES,
    NO_PRINT,
    create_arrays,
    DTYPE,
    write_results_to_file,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
)

LIB_NAME = "pycuda"


class PyCudaImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        drv.init()
        drv.Device(0).make_context()
        self.lib_name = LIB_NAME

    def warm_up(self):
        """
        Give CUDA an opportunity to compile these functions.
        """
        warm_up_arrays = [
            gpuarray.to_gpu(cpu_array)
            for cpu_array in create_arrays((1, 1, 1), self.dtype)
        ]
        self.add_arrays(*warm_up_arrays[:2])
        self.background_correction(*warm_up_arrays)

    def free_memory_pool(self):
        for gpu_array in self.gpu_arrays:
            gpu_array.gpudata.free()

    def _send_arrays_to_gpu(self):
        self.gpu_arrays = [gpuarray.to_gpu(np_arr) for np_arr in self.cpu_arrays]

    def time_function(self, func):
        drv.Context.synchronize()
        start = time.time()
        func()
        drv.Context.synchronize()
        end = time.time()
        return end - start

    @staticmethod
    def add_arrays(arr1, arr2):
        arr1 += arr2

    def timed_add_arrays(self, runs):
        operation_time = 0
        transfer_time = self.time_function(self._send_arrays_to_gpu)
        arr1, arr2 = self.gpu_arrays[:2]

        for _ in range(runs):
            operation_time += self.time_function(lambda: self.add_arrays(arr1, arr2))

        transfer_time += self.time_function(arr1.get)
        self.print_operation_times(
            operation_time / runs, ADD_ARRAYS, runs, transfer_time
        )

        return transfer_time + operation_time / runs

    @staticmethod
    def background_correction(data, dark, flat):
        norm_divide = flat - dark
        data -= dark
        flat -= dark
        data /= flat

    def timed_background_correction(self, runs):
        operation_time = 0
        transfer_time = self.time_function(self._send_arrays_to_gpu)
        data, dark, flat = self.gpu_arrays

        for _ in range(runs):
            operation_time += self.time_function(
                lambda: self.background_correction(data, dark, flat)
            )

        transfer_time += self.time_function(data.get)
        self.print_operation_times(
            operation_time / runs, BACKGROUND_CORRECTION, runs, transfer_time
        )

        return transfer_time + operation_time / runs


def print_memory_metrics():
    if NO_PRINT:
        return
    free, total = drv.mem_get_info()
    print("Used bytes:", total - free, "/ Total bytes:", total)


add_arrays = []
background_correction = []

for size in ARRAY_SIZES:

    try:

        imaging_obj = PyCudaImplementation(size, DTYPE)
        avg_add = imaging_obj.timed_add_arrays(20)
        avg_bc = imaging_obj.timed_background_correction(20)
        imaging_obj.free_memory_pool()
        del imaging_obj

    except (cp.cuda.memory.OutOfMemoryError, drv.MemoryError) as e:
        print(e)
        print("Unable to make GPU arrays with size", size)
        break

    add_arrays.append(avg_add)
    background_correction.append(avg_bc)

drv.Context.pop()

write_results_to_file([LIB_NAME, ADD_ARRAYS], add_arrays)
write_results_to_file([LIB_NAME, BACKGROUND_CORRECTION], background_correction)
