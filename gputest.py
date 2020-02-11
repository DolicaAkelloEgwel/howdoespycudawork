import cupy as cp
import numpy as np
import time

from cupy.cuda.stream import Event
from matplotlib import pyplot as plt
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv

from imagingtester import ImagingTester, NumpyImplementation, ARRAY_SIZES, TOTAL_PIXELS


class CupyImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)

    def _send_arrays_to_gpu(self):
        self.gpu_arrays = [cp.asarray(cpu_array) for cpu_array in self.cpu_arrays]

    @staticmethod
    def time_function(func):
        start = time.time()
        func()
        cp.cuda.runtime.deviceSynchronize()
        return time.time() - start

    def timed_add_arrays(self, runs):
        operation_time = 0

        transfer_time = self.time_function(self._send_arrays_to_gpu)
        arr1, arr2 = self.gpu_arrays[:2]

        for _ in range(runs):
            operation_time += self.time_function(lambda: cp.add(arr1, arr2))

        transfer_time += self.time_function(arr1.get)

        print(
            "With cupy transferring arrays of size %s took %ss and adding arrays took an average of %ss"
            % (self.cpu_arrays[0].shape, transfer_time, operation_time / runs)
        )

        return transfer_time + operation_time / runs

    def timed_background_correction(self, runs):
        operation_time = 0

        transfer_time = self.time_function(self._send_arrays_to_gpu)
        data, dark, flat = self.gpu_arrays

        def background_correction(data, dark, flat):
            cp.subtract(data, dark, out=data)
            cp.subtract(flat, dark, out=flat)
            cp.true_divide(data, flat, out=data)

        for _ in range(runs):
            operation_time += self.time_function(
                lambda: background_correction(data, dark, flat)
            )

        transfer_time += self.time_function(data.get)

        print(
            "With cupy transferring arrays of size %s took %ss and background correction took an average of %ss"
            % (self.cpu_arrays[0].shape, transfer_time, operation_time / runs)
        )

        return transfer_time + operation_time / runs


class PyCudaImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)

    def _send_arrays_to_gpu(self):
        self.gpu_arrays = [gpuarray.to_gpu(np_arr) for np_arr in self.cpu_arrays]

    def time_function(self, func):
        start = time.time()
        func()
        end = time.time()
        return end - start

    def timed_add_arrays(self, runs):
        operation_time = 0
        transfer_time = self.time_function(self._send_arrays_to_gpu)
        arr1, arr2 = self.gpu_arrays[:2]

        def add_arrays(arr1, arr2):
            arr1 += arr2

        for _ in range(runs):
            operation_time += self.time_function(lambda: add_arrays(arr1, arr2))

        transfer_time += self.time_function(arr1.get)

        print(
            "With pycuda transferring arrays of size %s took %ss and adding arrays took an average of %ss"
            % (self.cpu_arrays[0].shape, transfer_time, operation_time / runs)
        )

        return transfer_time + operation_time / runs

    def timed_background_correction(self, runs):
        operation_time = 0
        transfer_time = self.time_function(self._send_arrays_to_gpu)
        data, dark, flat = self.gpu_arrays

        def background_correction(data, dark, flat):
            data -= dark
            flat -= dark
            data /= flat

        for _ in range(runs):
            operation_time += self.time_function(
                lambda: background_correction(data, dark, flat)
            )

        transfer_time += self.time_function(data.get)

        print(
            "Wth pycuda transferring arrays of size %s took %ss and operation took an average of %ss"
            % (self.cpu_arrays[0].shape, transfer_time, operation_time / runs)
        )

        return transfer_time + operation_time / runs


# Create a dictionary for storing the run results
implementations = [CupyImplementation, NumpyImplementation, PyCudaImplementation]
results = {impl: dict() for impl in implementations}
function_names = ["Add Arrays", "Background Correction"]

use_mempool = True

if use_mempool:
    mempool = cp.get_default_memory_pool()

    with cp.cuda.Device(0):
        mempool.set_limit(fraction=1)

else:
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)


def clear_memory_pool(imaging_obj):

    if isinstance(imaging_obj, CupyImplementation):
        del imaging_obj
        if use_mempool:
            mempool.free_all_blocks()
    elif isinstance(imaging_obj, PyCudaImplementation):
        for gpu_array in imaging_obj.gpu_arrays:
            gpu_array.gpudata.free()
        del imaging_obj


def print_memory_metrics(ExecutionClass):

    if ExecutionClass is CupyImplementation and use_mempool:
        print(
            "Used bytes:", mempool.used_bytes(), "/ Total bytes:", mempool.total_bytes()
        )
    elif ExecutionClass is PyCudaImplementation:
        free, total = drv.mem_get_info()
        print("Used bytes:", total - free, "/ Total bytes:", total)


# Loop through the different libraries
for ExecutionClass in implementations:

    # Create empty lists for the results
    results[ExecutionClass]["Add Arrays"] = []
    results[ExecutionClass]["Background Correction"] = []

    # Loop through the different array sizes
    for size in ARRAY_SIZES:

        try:

            imaging_obj = ExecutionClass(size)
            # warm up
            imaging_obj.timed_add_arrays(1)
            avg_add = imaging_obj.timed_add_arrays(20)
            print_memory_metrics(ExecutionClass)
            clear_memory_pool(imaging_obj)

            imaging_obj = ExecutionClass(size)
            # warm up
            imaging_obj.timed_background_correction(1)
            avg_bc = imaging_obj.timed_background_correction(20)
            print_memory_metrics(ExecutionClass)
            clear_memory_pool(imaging_obj)

        except (cp.cuda.memory.OutOfMemoryError, drv.MemoryError) as e:
            print(e)
            print("Unable to make GPU arrays with size", size)
            print_memory_metrics(ExecutionClass)
            break

        results[ExecutionClass]["Add Arrays"].append(avg_add)
        results[ExecutionClass]["Background Correction"].append(avg_bc)


library_labels = {
    CupyImplementation: "cupy",
    NumpyImplementation: "numpy",
    PyCudaImplementation: "pycuda",
}

## Plot adding times
plt.subplot(2, 2, 1)
plt.title("Average Time Taken To Add Two Arrays")

for impl in implementations:
    plt.plot(results[impl]["Add Arrays"], label=library_labels[impl], marker=".")

plt.ylabel("Time Taken")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.yscale("log")
plt.legend()

## Plot Background Correction Times
plt.subplot(2, 2, 3)
plt.title("Average Time Taken To Do Background Correction")

for impl in implementations:
    plt.plot(
        results[impl]["Background Correction"], label=library_labels[impl], marker="."
    )

plt.ylabel("Time Taken")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.yscale("log")
plt.xlabel("Number of Pixels/Elements")

## Plot speed-up for cupy
ax = plt.subplot(2, 2, 2)
plt.title("Speed Boost Obtained From Using cupy Over numpy")
ax.set_prop_cycle(color=["purple", "red"])

# Determine the speed up by diving numpy time by gpu time and plot
for func in function_names:
    speed_up = np.divide(
        results[NumpyImplementation][func][: len(results[CupyImplementation][func])],
        results[CupyImplementation][func],
    )
    plt.plot(speed_up, label=func, marker=".")

plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.legend()
plt.ylabel("Avg np Time / Avg cp Time")

## Plot speed-up for pycuda
ax = plt.subplot(2, 2, 4)
plt.title("Speed Boost Obtained From Using pycuda Over numpy")
ax.set_prop_cycle(color=["black", "yellow"])

# Determine the speed up by diving numpy time by gpu time and plot
for func in function_names:
    speed_up = np.divide(
        results[NumpyImplementation][func][: len(results[PyCudaImplementation][func])],
        results[PyCudaImplementation][func],
    )
    plt.plot(speed_up, label=func, marker=".")

plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.legend()
plt.xlabel("Number of Pixels/Elements")
plt.ylabel("Avg np Time / Avg pycuda Time")

print(results[CupyImplementation]["Background Correction"])
print(results[PyCudaImplementation]["Background Correction"])

plt.show()
