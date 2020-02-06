import cupy as cp
import numpy as np
import time
from matplotlib import pyplot as plt

class ImagingTester:
    def __init__(self):
        self.arrays = None
    def create_arrays(self, num_arrays, size_tuple):
        self.arrays = [np.random.rand(*size_tuple) for _ in range(num_arrays)]
    def add_arrays(self):
        pass
    def background_correction(self):
        pass
        
class NumpyImplemention(ImagingTester):
    def __init__(self):
        super().__init__() 
    def add_arrays(self):
        return np.add(*self.arrays)
    def background_correction(self):
        np.subtract(self.arrays[0], self.arrays[1], self.arrays[0])
        np.subtract(self.arrays[2], self.arrays[1], out=self.arrays[2])
        np.true_divide(self.arrays[0], self.arrays[2], out=self.arrays[0])
        return self.arrays[0]
        
class CupyImplementation(ImagingTester):
    def __init__(self):
        super().__init__()
    def _send_arrays_to_gpu(self):
        self.arrays = [cp.asarray(np_arr) for np_arr in self.arrays]
    def add_arrays(self):
        self._send_arrays_to_gpu()
        return cp.add(*self.arrays).get()
    def background_correction(self):
        self._send_arrays_to_gpu()
        cp.subtract(self.arrays[0], self.arrays[1], self.arrays[0])
        cp.subtract(self.arrays[2], self.arrays[1], out=self.arrays[2])
        cp.true_divide(self.arrays[0], self.arrays[2], out=self.arrays[0])
        return self.arrays[0].get()

# Create a function for timing imaging-related operations
def cool_timer(imaging_obj, size, num_arrs, imaging_alg):
    imaging_obj.create_arrays(num_arrs, size)
    if imaging_alg == "Add Arrays":
        start = time.time()
        imaging_obj.add_arrays()
        end = time.time()
    else:
        start = time.time()
        imaging_obj.background_correction()
        end = time.time()
    return end - start

# Create lists of array sizes and the total number of pixels/elements
array_sizes = [(10, 100), (100, 100), (100, 1000), (1000, 1000), (1000, 2000), (2000, 2000)]
total_pixels = [x[0] * x[1] for x in array_sizes]

# Create a dictionary for storing the run results
results = {NumpyImplemention: dict(), CupyImplementation: dict()}

# Loop through the different libraries
for ExecutionClass in [NumpyImplemention, CupyImplementation]:

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
            total_add += cool_timer(imaging_obj, size, 2, "Add Arrays")
            total_bc += cool_timer(imaging_obj, size, 3, "Background Correction")

        # Compute the average speed for the 10 runs
        results[ExecutionClass]["Add Arrays"].append(total_add / 10)
        results[ExecutionClass]["Background Correction"].append(total_bc / 10)
        
print(results)

library_labels = {CupyImplementation: "cupy", NumpyImplemention: "numpy", }

## Plot adding times
plt.subplot(2, 2, 1)
plt.title("Average Time Taken To Add Two Arrays")

for impl in [CupyImplementation, NumpyImplemention]:
    plt.plot(results[impl]["Add Arrays"], label=library_labels[impl], marker=".")

plt.ylabel("Time Taken")
plt.xticks(range(len(total_pixels)), total_pixels)
plt.yscale("log")
plt.legend()

## Plot Background Correction Times
#plt.subplot(2, 2, 2)
#plt.title("Average Time Taken To Do Background Correction")

#for lib in [cp, np]:
#    plt.plot(results[lib][background_correction_test], label=library_labels[lib], marker=".")

#plt.ylabel("Time Taken")
#plt.xticks(range(len(total_pixels)), total_pixels)
#plt.yscale("log")

## Plot speed-up
#ax = plt.subplot(2, 2, 3)
#plt.title("Speed Boost Obtained From Using cupy Over numpy")
#ax.set_prop_cycle(color=['purple', 'green'])

## Determine the speed up by diving numpy time by gpu time and plot
#for func in funcs:
#    speed_up = np.divide(results[np][func], results[cp][func])
#    plt.plot(speed_up, label=function_labels[func], marker=".")

#plt.xticks(range(len(total_pixels)), total_pixels)
#plt.legend()
#plt.ylabel("Avg np Time / Avg cp Time")
#plt.xlabel("Number of Pixels/Elements")

plt.show()

# Just seeing if PyCUDA was installed
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

a_gpu = gpuarray.to_gpu(numpy.random.randn(4, 4).astype(numpy.float32))
a_doubled = (a_gpu * a_gpu).get()
print(a_doubled)
print(a_gpu)
