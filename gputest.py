import cupy as cp
import numpy as np
import time

def cool_timer(func, lib, size, num_arrs):
    arrays = create_arrays(lib, num_arrs)
    start = time.time()
    func(lib, *arrays)
    end = time.time()
    print(end - start)

def create_arrays(matrix_library, num_arrs):
    return [matrix_library.random.rand(size, size) for _ in range(num_arrs)]

def add_arrays(matrix_library, first_array, second_array):
    matrix_library.add(first_array, second_array)
    
def background_correction_test(matrix_library, data, dark, flat):
    matrix_library.subtract(data, dark, out=data)
    matrix_library.subtract(flat, dark, out=flat)
    matrix_library.true_divide(data, flat, out=data)
    
array_sizes = [100, 1000, 5000, 10000, 20000]

for size in array_sizes:
    for lib in [cp, np]:
        cool_timer(add_arrays, lib, size, 2)
        cool_timer(background_correction_test, lib, size, 3)
        print("")
