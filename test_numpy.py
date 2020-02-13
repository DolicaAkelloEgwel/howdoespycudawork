import time

import numpy as np

from imagingtester import (
    ImagingTester,
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
    ARRAY_SIZES,
    N_RUNS,
    write_results_to_file,
    SIZES_SUBSET,
    DTYPE,
)

LIB_NAME = "numpy"


class NumpyImplementation(ImagingTester):
    def __init__(self, size, dtype="float32"):
        super().__init__(size, dtype)
        self.lib_name = LIB_NAME

    @staticmethod
    def time_function(func):
        start = time.time()
        func()
        end = time.time()
        return end - start

    def timed_add_arrays(self, reps):
        total_time = 0
        for _ in range(reps):
            total_time += self.time_function(lambda: np.add(*self.cpu_arrays[:2]))
        operation_time = total_time / reps
        self.print_operation_times(operation_time, "adding", reps, None)
        return operation_time

    @staticmethod
    def background_correction(dark, data, flat):
        norm_divide = np.subtract(flat, dark)
        norm_divide[norm_divide == 0] = MINIMUM_PIXEL_VALUE
        np.subtract(data, dark, out=data)
        np.true_divide(data, norm_divide, out=data)
        np.clip(data, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE, out=data)

    def timed_background_correction(self, reps):
        data, dark, flat = self.cpu_arrays
        total_time = 0
        for _ in range(reps):
            total_time += self.time_function(
                lambda: self.background_correction(dark, data, flat)
            )
        operation_time = total_time / reps
        self.print_operation_times(operation_time, "background correction", reps, None)
        return operation_time


# Create empty lists for storing results
add_arrays = []
background_correction = []

for size in ARRAY_SIZES[:SIZES_SUBSET]:

    start = time.time()
    imaging_obj = NumpyImplementation(size, DTYPE)
    end = time.time()
    print("Array creation time", end - start)

    add_arrays.append(imaging_obj.timed_add_arrays(N_RUNS))
    background_correction.append(imaging_obj.timed_background_correction(N_RUNS))

write_results_to_file([LIB_NAME, "add arrays"], add_arrays)
write_results_to_file([LIB_NAME, "background correction"], background_correction)
