import cupy as cp
import numpy as np
import timeit
from matplotlib import pyplot as plt
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from numba import jit

MINIMUM_PIXEL_VALUE = 1e-9
MAXIMUM_PIXEL_VALUE = 1e9


class ImagingTester:
    def __init__(self, size):
        self.create_arrays(size)

    def create_arrays(self, size_tuple):
        self.arrays = [
            np.random.uniform(
                low=MINIMUM_PIXEL_VALUE, high=MAXIMUM_PIXEL_VALUE, size=size_tuple
            )
            for _ in range(3)
        ]

    def add_arrays(self):
        pass

    def background_correction(self):
        pass


class NumpyImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)

    def add_arrays(self, arr1, arr2):
        np.add(arr1, arr2)

    def background_correction(self, data, dark, flat):
        np.subtract(data, dark, out=data)
        np.subtract(flat, dark, out=flat)
        np.true_divide(data, flat, out=data)


class CupyImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)
        self._send_arrays_to_gpu()

    def _send_arrays_to_gpu(self):
        self.arrays = [cp.asarray(np_arr) for np_arr in self.arrays]

    def add_arrays(self, arr1, arr2):
        cp.add(arr1, arr2)

    def background_correction(self, data, dark, flat):
        cp.subtract(data, dark, out=data)
        cp.subtract(flat, dark, out=flat)
        cp.true_divide(data, flat, out=data)


class PyCudaImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)
        self._send_arrays_to_gpu()

    def _send_arrays_to_gpu(self):
        self.arrays = [gpuarray.to_gpu(np_arr) for np_arr in self.arrays]

    def add_arrays(self, arr1, arr2):
        arr1 + arr2

    def background_correction(self, data, dark, flat):
        data - dark
        flat - dark
        data / flat


class NumbaImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)

    @staticmethod
    @jit("void(float64[:,:],float64[:,:])", nopython=True)
    def add_arrays(arr1, arr2):
        for i in range(len(arr1)):
            for j in range(len(arr1[0])):
                arr1[i][j] += arr2[i][j]

    @staticmethod
    @jit("void(float64[:,:],float64[:,:],float64[:,:])", nopython=True)
    def background_correction(data, dark, flat):
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] -= dark[i][j]
                flat[i][j] -= dark[i][j]
                data[i][j] /= flat[i][j]


# Create a function for timing imaging-related operations
def cool_timer(imaging_obj, func):
    return timeit.timeit(func, number=20)


# Create lists of array sizes and the total number of pixels/elements
array_sizes = [
    (10, 100),
    (100, 100),
    (100, 1000),
    (1000, 1000),
    (1000, 2000),
    (2000, 2000),
    (2500, 2500),
    (3000, 3000),
    (4000, 4000),
]
total_pixels = [x[0] * x[1] for x in array_sizes]

# Create a dictionary for storing the run results
implementations = [
    PyCudaImplementation,
    NumpyImplementation,
    CupyImplementation,
    NumbaImplementation,
]
results = {impl: dict() for impl in implementations}
function_names = ["Add Arrays", "Background Correction"]

# Loop through the different libraries
for ExecutionClass in implementations:

    # Create empty lists for the results
    results[ExecutionClass]["Add Arrays"] = []
    results[ExecutionClass]["Background Correction"] = []

    # Loop through the different array sizes
    for size in array_sizes:

        total_add = 0
        total_bc = 1

        imaging_obj = ExecutionClass(size)

        # Run the functions for the current array size 10 times
        total_add = cool_timer(
            imaging_obj, lambda: imaging_obj.add_arrays(*imaging_obj.arrays[:2])
        )
        total_bc = cool_timer(
            imaging_obj, lambda: imaging_obj.background_correction(*imaging_obj.arrays)
        )

        assert size == imaging_obj.arrays[0].shape

        results[ExecutionClass]["Add Arrays"].append(total_add)
        results[ExecutionClass]["Background Correction"].append(total_bc)

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
        results[NumpyImplementation][func], results[CupyImplementation][func]
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
        results[NumpyImplementation][func], results[PyCudaImplementation][func]
    )
    plt.plot(speed_up, label=func, marker=".")

plt.xticks(range(len(total_pixels)), total_pixels)
plt.legend()
plt.xlabel("Number of Pixels/Elements")
plt.ylabel("Avg np Time / Avg pycuda Time")

plt.show()
