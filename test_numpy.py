import time

import numpy as np

from imagingtester import ImagingTester, SIZES_SUBSET, DTYPE, N_RUNS
from numpy_background_correction import numpy_background_correction
from write_and_read_results import (
    write_results_to_file,
    ARRAY_SIZES,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
)

LIB_NAME = "numpy"


def time_function(func):
    start = time.time()
    func()
    end = time.time()
    return end - start


class NumpyImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.lib_name = LIB_NAME

    def timed_add_arrays(self, reps):
        total_time = 0
        for _ in range(reps):
            total_time += time_function(lambda: np.add(*self.cpu_arrays[:2]))
        operation_time = total_time / reps
        self.print_operation_times(
            operation_time=operation_time, operation_name="adding", runs=reps
        )
        return operation_time

    def timed_background_correction(self, reps):
        data, dark, flat = self.cpu_arrays
        total_time = 0
        for _ in range(reps):
            total_time += time_function(
                lambda: numpy_background_correction(dark, data, flat)
            )
        operation_time = total_time / reps
        self.print_operation_times(
            operation_time=operation_time,
            operation_name="background correction",
            runs=reps,
        )
        return operation_time


# Create empty lists for storing results
add_arrays_results = []
background_correction_results = []

for size in ARRAY_SIZES[:SIZES_SUBSET]:

    imaging_obj = NumpyImplementation(size, DTYPE)

    add_arrays_results.append(imaging_obj.timed_add_arrays(N_RUNS))
    background_correction_results.append(
        imaging_obj.timed_background_correction(N_RUNS)
    )

write_results_to_file([LIB_NAME], ADD_ARRAYS, add_arrays_results)
write_results_to_file([LIB_NAME], BACKGROUND_CORRECTION, background_correction_results)
