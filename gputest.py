import cupy as cp
import numpy as np
import time
from matplotlib import pyplot as plt
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit


class ImagingTester:
    def __init__(self):
        self.arrays = None

    def create_arrays(self, num_arrays, size_tuple):
        self.arrays = [np.random.rand(*size_tuple) for _ in range(num_arrays)]

    def add_arrays(self):
        pass

    def background_correction(self):
        pass


class NumpyImplementation(ImagingTester):
    def __init__(self):
        super().__init__()

    def _send_arrays_to_gpu(self):
        pass

    def add_arrays(self):
        np.add(*self.arrays)

    def background_correction(self):
        np.subtract(self.arrays[0], self.arrays[1], out=self.arrays[0])
        np.subtract(self.arrays[2], self.arrays[1], out=self.arrays[2])
        np.true_divide(self.arrays[0], self.arrays[2], out=self.arrays[0])


class CupyImplementation(ImagingTester):
    def __init__(self):
        super().__init__()

    def _send_arrays_to_gpu(self):
        self.arrays = [cp.asarray(np_arr) for np_arr in self.arrays]

    def add_arrays(self):
        cp.add(*self.arrays)

    def background_correction(self):
        cp.subtract(self.arrays[0], self.arrays[1], out=self.arrays[0])
        cp.subtract(self.arrays[2], self.arrays[1], out=self.arrays[2])
        cp.true_divide(self.arrays[0], self.arrays[2], out=self.arrays[0])


class PyCudaImplementation(ImagingTester):
    def __init__(self):
        super().__init__()

    def _send_arrays_to_gpu(self):
        self.arrays = [gpuarray.to_gpu(np_arr) for np_arr in self.arrays]

    def add_arrays(self):
        self.arrays[0] + self.arrays[1]

    def background_correction(self):
        self.arrays[0] - self.arrays[1]
        self.arrays[2] - self.arrays[1]
        self.arrays[0] / self.arrays[2]


# Create a function for timing imaging-related operations
def cool_timer(imaging_obj, size, num_arrs, func):
    imaging_obj.create_arrays(num_arrs, size)
    imaging_obj._send_arrays_to_gpu()
    start = time.time()
    func()
    end = time.time()
    return end - start


# Create lists of array sizes and the total number of pixels/elements
array_sizes = [
    (10, 100),
    (100, 100),
    (100, 1000),
    (1000, 1000),
    (1000, 2000),
    (2000, 2000),
]
total_pixels = [x[0] * x[1] for x in array_sizes]

# Create a dictionary for storing the run results
implementations = [CupyImplementation, NumpyImplementation, PyCudaImplementation]
results = {impl: dict() for impl in implementations}
function_names = ["Add Arrays", "Background Correction"]

# Loop through the different libraries
for ExecutionClass in implementations:

    imaging_obj = ExecutionClass()

    # Create empty lists for the results
    results[ExecutionClass]["Add Arrays"] = []
    results[ExecutionClass]["Background Correction"] = []

    # Loop through the different array sizes
    for size in array_sizes:

        total_add = 0
        total_bc = 0

        # Run the functions for the current array size 10 times
        for _ in range(10):
            total_add += cool_timer(imaging_obj, size, 2, imaging_obj.add_arrays)
            total_bc += cool_timer(
                imaging_obj, size, 3, imaging_obj.background_correction
            )

        # Compute the average speed for the 10 runs
        results[ExecutionClass]["Add Arrays"].append(total_add / 10)
        results[ExecutionClass]["Background Correction"].append(total_bc / 10)

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
        results[NumpyImplementation][func], results[CupyImplementation][func]
    )
    plt.plot(speed_up, label=func, marker=".")

plt.xticks(range(len(total_pixels)), total_pixels)
plt.legend()
plt.ylabel("Avg np Time / Avg cp Time")

## Plot speed-up for pycuda
ax = plt.subplot(2, 2, 4)
plt.title("Speed Boost Obtained From Using cupy Over numpy")
ax.set_prop_cycle(color=["black", "yellow"])

# Determine the speed up by diving numpy time by gpu time and plot
for func in function_names:
    speed_up = np.divide(
        results[NumpyImplementation][func], results[PyCudaImplementation][func]
    )
    plt.plot(speed_up, label=func, marker=".")

plt.xticks(range(len(total_pixels)), total_pixels)
plt.legend()
plt.xlabel("Number of Pixels/Elements")
plt.ylabel("Avg np Time / Avg pycuda Time")

plt.show()
