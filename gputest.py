import cupy as cp
import numpy as np
import time
import matplotlib

def cool_timer(func, lib, size, num_arrs):
    arrays = create_arrays(lib, num_arrs)
    start = time.time()
    func(lib, *arrays)
    end = time.time()
    return end - start

def create_arrays(matrix_library, num_arrs):
    return [matrix_library.random.rand(size, size) for _ in range(num_arrs)]

def add_arrays(matrix_library, first_array, second_array):
    matrix_library.add(first_array, second_array)
    
def background_correction_test(matrix_library, data, dark, flat):
    matrix_library.subtract(data, dark, out=data)
    matrix_library.subtract(flat, dark, out=flat)
    matrix_library.true_divide(data, flat, out=data)
    
array_sizes = [100, 1000, 5000, 10000, 20000]

names = {cp: "cupy", np: "numpy"}

for size in array_sizes:
    for lib in [cp, np]:
        total_add = 0
        total_bc = 0
        for _ in range(10):
            total_add += cool_timer(add_arrays, lib, size, 2)
            total_bc += cool_timer(background_correction_test, lib, size, 3)
        print("Library:", names[lib], "/ Size:", size)
        print(total_add / 10)
        print(total_bc / 10)
        print("")
