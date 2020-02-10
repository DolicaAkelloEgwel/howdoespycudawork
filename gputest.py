import cupy as cp
import numpy as np
import time

from cupy.cuda.stream import Event
from matplotlib import pyplot as plt
import pycuda.gpuarray as gpuarray
from numba import jit
import pycuda.driver as drv
import pycuda.autoinit  # MUST BE IMPORTED FOR PYCUDA TIMING TO WORK

MINIMUM_PIXEL_VALUE = 1e-9
MAXIMUM_PIXEL_VALUE = 1e9


class ImagingTester:
    def __init__(self, size):
        self.create_arrays(size)

    def create_arrays(self, size_tuple):
        self.cpu_arrays = [
            np.random.uniform(
                low=MINIMUM_PIXEL_VALUE, high=MAXIMUM_PIXEL_VALUE, size=size_tuple
            ).astype("float32")
            for _ in range(3)
        ]

    def timed_add_arrays(self):
        pass

    def timed_background_correction(self):
        pass


class NumpyImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)

    def timed_add_arrays(self, reps):
        arr1, arr2 = self.cpu_arrays[:2]
        total_time = 0
        for _ in range(reps):
            start = time.time()
            ###
            np.add(arr1, arr2)
            ###
            total_time += time.time() - start
        return total_time / reps

    def timed_background_correction(self, reps):
        data, dark, flat = self.cpu_arrays
        total_time = 0
        for _ in range(reps):
            start = time.time()
            ###
            np.subtract(data, dark, out=data)
            np.subtract(flat, dark, out=flat)
            np.true_divide(data, flat, out=data)
            ###
            total_time += time.time() - start
        return total_time / reps


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
            "Transferring took %ss and operation took an average of %ss"
            % (transfer_time, operation_time / runs)
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
            "Transferring took %ss and operation took an average of %ss"
            % (transfer_time, operation_time / runs)
        )

        return transfer_time + operation_time / runs


class PyCudaImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)

    def _send_arrays_to_gpu(self):
        self.gpu_arrays = [gpuarray.to_gpu(np_arr) for np_arr in self.cpu_arrays]

    def time_function(self, func):
        start = drv.Event()
        end = drv.Event()
        start.record()
        start.synchronize()
        func()
        end.record()
        end.synchronize()
        return start.time_till(end) * 1e3

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
            "Transferring took %ss and operation took an average of %ss"
            % (transfer_time, operation_time / runs)
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
            "Transferring took %ss and operation took an average of %ss"
            % (transfer_time, operation_time / runs)
        )

        return transfer_time + operation_time / runs


class NumbaImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)

    @staticmethod
    @jit("void(float64[:,:],float64[:,:])", nopython=True)
    def timed_add_arrays(arr1, arr2):
        for i in range(len(arr1)):
            for j in range(len(arr1[0])):
                arr1[i][j] += arr2[i][j]

    @staticmethod
    @jit("void(float64[:,:],float64[:,:],float64[:,:])", nopython=True)
    def timed_background_correction(data, dark, flat):
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] -= dark[i][j]
                flat[i][j] -= dark[i][j]
                data[i][j] /= flat[i][j]


# Create a function for timing imaging-related operations
def cool_timer(func):
    total_time = 0
    total_runs = 20
    for _ in range(total_runs):
        cp.cuda.runtime.deviceSynchronize()
        start = time.time()
        func()
        cp.cuda.runtime.deviceSynchronize()
        total_time += time.time() - start
    return total_time / total_runs


# Create lists of array sizes and the total number of pixels/elements
array_sizes = [
    (10, 100, 500),
    (100, 100, 500),
    (100, 1000, 500),
    (1000, 1000, 500),
    (1500, 1500, 500),
]
total_pixels = [x * y * z for x, y, z in array_sizes]

# Create a dictionary for storing the run results
implementations = [
    CupyImplementation,
    NumpyImplementation,
    PyCudaImplementation,
    # NumbaImplementation,
]
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
    for size in array_sizes:

        total_add = 0
        total_bc = 1

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
    NumbaImplementation: "numba",
}

## Plot adding times
plt.subplot(2, 2, 1)
plt.title("Average Time Taken To Add Two Arrays")

for impl in implementations:
    plt.plot(results[impl]["Add Arrays"], label=library_labels[impl], marker=".")

plt.ylabel("Time Taken")
plt.xticks(range(len(total_pixels)), total_pixels)
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
plt.xticks(range(len(total_pixels)), total_pixels)
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

plt.xticks(range(len(total_pixels)), total_pixels)
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

plt.xticks(range(len(total_pixels)), total_pixels)
plt.legend()
plt.xlabel("Number of Pixels/Elements")
plt.ylabel("Avg np Time / Avg pycuda Time")

print(results[CupyImplementation]["Background Correction"])
print(results[PyCudaImplementation]["Background Correction"])

plt.show()
