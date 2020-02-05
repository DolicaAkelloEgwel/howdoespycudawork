import cupy as cp
import numpy as np
import time
from matplotlib import pyplot as plt

# Create a function for timing imaging-related operations
def cool_timer(func, lib, size, num_arrs):
    arrays = create_arrays(lib, size, num_arrs)
    start = time.time()
    func(lib, *arrays)
    end = time.time()
    return end - start

# Create a given number of arrays
def create_arrays(matrix_library, size_tuple, num_arrs):
    return [matrix_library.random.rand(*size_tuple) for _ in range(num_arrs)]

# Add two arrays 
def add_arrays(matrix_library, first_array, second_array):
    matrix_library.add(first_array, second_array)
    
# Do something akin to background correction with made-up data
def background_correction_test(matrix_library, data, dark, flat):
    matrix_library.subtract(data, dark, out=data)
    matrix_library.subtract(flat, dark, out=flat)
    matrix_library.true_divide(data, flat, out=data)

# Create a list of functions used in the parallel comparison test
funcs = [add_arrays, background_correction_test]

# Create lists of array sizes and the total number of pixels/elements
array_sizes = [(10,100), (100,100), (100,1000), (1000,1000), (1000,2000), (2000,2000)]
total_pixels = [x[0] * x[1] for x in array_sizes]

# Create a dictionary for storing the run results
results = {cp: dict(), np: dict()}

# Loop through the different libraries
for lib in [cp, np]:

    # Create empty lists for the results
    results[lib][add_arrays] = []
    results[lib][background_correction_test] = []
    
    # Loop through the different array sizes
    for size in array_sizes:
        total_add = 0
        total_bc = 0
        
        # Run the functions for the current array size 10 times
        for _ in range(10):
            total_add += cool_timer(add_arrays, lib, size, 2)
            total_bc += cool_timer(background_correction_test, lib, size, 3)
            
        # Compute the average speed for the 10 runs
        results[lib][add_arrays].append(total_add / 10)
        results[lib][background_correction_test].append(total_bc / 10)

labels = {add_arrays: "Add Arrays", background_correction_test: "Background Correction", }

# Determine the speed up by diving numpy time by gpu time
for func in funcs:
    speed_up = np.divide(results[np][func], results[cp][func])
    plt.plot(speed_up,label=labels[func], marker=".")

# Plot speed up
plt.xticks(range(len(total_pixels)), total_pixels)
plt.legend()
plt.xlabel("Number of Elements")
plt.ylabel("Average Time Improvement (Numpy Time / Cupy Time)")
plt.show()
