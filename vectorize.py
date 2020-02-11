from numba import vectorize
import numba
import numpy as np
import time
from matplotlib import pyplot as plt

from imagingtester import ImagingTester, NumpyImplementation, ARRAY_SIZES, TOTAL_PIXELS


@vectorize(["float32(float32, float32)"], target="cuda")
def cuda_add_arrays(elem1, elem2):
    return elem1 + elem2


@vectorize("float32(float32,float32,float32)", target="cuda")
def cuda_background_correction(data, dark, flat):
    data -= dark
    flat -= dark
    if flat != 0:
        return data / flat
    return data


@vectorize(["float32(float32, float32)"], nopython=True, target="parallel")
def parallel_add_arrays(elem1, elem2):
    return elem1 + elem2


@vectorize("float32(float32,float32,float32)", nopython=True, target="parallel")
def parallel_background_correction(data, dark, flat):
    data -= dark
    flat -= dark
    if flat != 0:
        return data / flat
    return data


class CudaNumbaImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)

    def time_function(self, func):
        start = time.time()
        func()
        return time.time() - start

    def timed_add_arrays(self, runs):
        total_time = 0

        for _ in range(runs):
            total_time += self.time_function(
                lambda: cuda_add_arrays(*self.cpu_arrays[:2])
            )
        return total_time / runs

    def timed_background_correction(self, runs):
        total_time = 0

        for _ in range(runs):
            total_time += self.time_function(
                lambda: cuda_background_correction(*self.cpu_arrays)
            )
        return total_time


class ParallelNumbaImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)

    def time_function(self, func):
        start = time.time()
        func()
        return time.time() - start

    def timed_add_arrays(self, runs):
        total_time = 0

        for _ in range(runs):
            total_time += self.time_function(
                lambda: parallel_add_arrays(*self.cpu_arrays[:2])
            )
        return total_time / runs

    def timed_background_correction(self, runs):
        total_time = 0

        for _ in range(runs):
            total_time += self.time_function(
                lambda: parallel_background_correction(*self.cpu_arrays)
            )
        return total_time


# Warm up
warm_up_arrays = [
    np.random.randint(0, 100, size=(5, 5, 5)).astype("float32") for _ in range(3)
]
cuda_add_arrays(*warm_up_arrays[:2])
parallel_add_arrays(*warm_up_arrays[:2])
cuda_background_correction(*warm_up_arrays)
parallel_background_correction(*warm_up_arrays)
print("Warm up complete.")

implementations = [
    NumpyImplementation,
    ParallelNumbaImplementation,
    CudaNumbaImplementation,
]
results = {impl: dict() for impl in implementations}
labels = {
    NumpyImplementation: "numpy",
    CudaNumbaImplementation: "numba + cuda",
    ParallelNumbaImplementation: "parallel numba",
}
function_names = ["Add Arrays", "Background Correction"]

for impl in implementations:

    # Create empty lists for the results
    results[impl]["Add Arrays"] = []
    results[impl]["Background Correction"] = []

    for size in ARRAY_SIZES:

        try:
            imaging_obj = impl(size)

            avg_add = imaging_obj.timed_add_arrays(20)
            avg_bc = imaging_obj.timed_background_correction(20)

            results[impl]["Add Arrays"].append(avg_add)
            results[impl]["Background Correction"].append(avg_bc)
        except numba.cuda.cudadrv.driver.CudaAPIError as e:
            print("Unable to carry out calculation on array of size", size)
            print(e)
            break

    print(results)

## Plot adding times
plt.subplot(2, 2, 1)
plt.title("Average Time Taken To Add Two Arrays")

for impl in implementations:
    plt.plot(results[impl]["Add Arrays"], label=labels[impl], marker=".")

plt.ylabel("Time Taken")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.yscale("log")
plt.legend()

## Plot Background Correction Times
plt.subplot(2, 2, 3)
plt.title("Average Time Taken To Do Background Correction")

for impl in implementations:
    plt.plot(results[impl]["Background Correction"], label=labels[impl], marker=".")

plt.ylabel("Time Taken")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.yscale("log")
plt.xlabel("Number of Pixels/Elements")

## Plot speed-up for parallel numba
ax = plt.subplot(2, 2, 2)
plt.title("Speed Boost Obtained From Using parallel numba Over numpy")
ax.set_prop_cycle(color=["purple", "red"])

# Determine the speed up by diving numpy time by gpu time and plot
for func in function_names:
    speed_up = np.divide(
        results[NumpyImplementation][func][
            : len(results[ParallelNumbaImplementation][func])
        ],
        results[ParallelNumbaImplementation][func],
    )
    plt.plot(speed_up, label=func, marker=".")

plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.legend()
plt.ylabel("Avg np Time / Avg parallel numba Time")

## Plot speed-up for pycuda
ax = plt.subplot(2, 2, 4)
plt.title("Speed Boost Obtained From Using cuda numba Over numpy")
ax.set_prop_cycle(color=["black", "yellow"])

# Determine the speed up by diving numpy time by gpu time and plot
for func in function_names:
    speed_up = np.divide(
        results[NumpyImplementation][func][
            : len(results[CudaNumbaImplementation][func])
        ],
        results[CudaNumbaImplementation][func],
    )
    plt.plot(speed_up, label=func, marker=".")

plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.legend()
plt.xlabel("Number of Pixels/Elements")
plt.ylabel("Avg np Time / Avg cuda numba Time")

plt.show()
