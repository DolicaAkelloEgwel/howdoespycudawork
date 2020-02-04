import cupy as cp
import numpy as np
import time

def cool_timer(func):
    start = time.time()
    func()
    end = time.time()
    print(end - start)


def add_arrays(matrix_library, size):
    first_array = matrix_library.random.rand(size, size)
    second_array = matrix_library.random.rand(size, size)
    matrix_library.add(first_array, second_array)
    
cool_timer(lambda: add_arrays(cp, 10000))
cool_timer(lambda: add_arrays(np, 10000))
