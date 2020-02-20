import time
import numpy as np

from imagingtester import (
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
    N_RUNS,
    DTYPE,
    create_arrays,
    SIZES_SUBSET,
    TEST_PARALLEL_NUMBA,
)
from numba_test_utils import (
    NumbaImplementation,
    create_vectorise_add_arrays,
    create_vectorise_background_correction,
    LIB_NAME,
)
from numpy_background_correction import numpy_background_correction
from write_and_read_results import (
    write_results_to_file,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
    ARRAY_SIZES,
)

mode = "parallel"

if not TEST_PARALLEL_NUMBA:
    exit()

add_arrays = create_vectorise_add_arrays(mode)
background_correction = create_vectorise_background_correction(mode)


class NumbaParallelImplementation(NumbaImplementation):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.warm_up()
        self.lib_name = LIB_NAME

    def warm_up(self):
        """
        Give the functions a chance to compile.
        """
        warm_up_arrays = create_arrays((1, 1, 1), self.dtype)
        add_arrays(*warm_up_arrays[:2])
        background_correction(*warm_up_arrays, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE)

    def get_time(self):
        return time.time()

    def timed_imaging_operation(self, runs, alg, alg_name, n_arrs_needed):

        operation_time = 0

        # Repeat the operation
        for _ in range(runs):
            operation_time += self.time_function(
                lambda: alg(*self.cpu_arrays[:n_arrs_needed])
            )

        self.print_operation_times(
            operation_time=operation_time, operation_name=alg_name, runs=runs
        )

        return operation_time / runs


practice_array = np.ones(shape=(5, 5, 5)).astype(DTYPE)
parallel_result = add_arrays(practice_array, practice_array)
assert np.all(parallel_result == 2)


# Checking the two background corrections get the same result
np_data, np_dark, np_flat = [
    np.random.uniform(low=0.0, high=20, size=(5, 5, 5)) for _ in range(3)
]

numba_data = np_data.copy()
numba_dark = np_dark.copy()
numba_flat = np_flat.copy()

background_correction(
    numba_data, numba_dark, numba_flat, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
)
numpy_background_correction(
    np_data, np_dark, np_flat, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
)
assert np.allclose(np_data, numba_data)


add_arrays_results = []
background_correction_results = []


def background_correction_fixed_clips(dark, data, flat):
    return background_correction(
        data, dark, flat, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
    )


for size in ARRAY_SIZES[:SIZES_SUBSET]:

    imaging_obj = NumbaParallelImplementation(size, DTYPE)

    avg_add = imaging_obj.timed_imaging_operation(N_RUNS, add_arrays, "adding", 2)
    avg_bc = imaging_obj.timed_imaging_operation(
        N_RUNS, background_correction_fixed_clips, "background correction", 3
    )

    add_arrays_results.append(avg_add)
    background_correction_results.append(avg_bc)

write_results_to_file([LIB_NAME, mode], ADD_ARRAYS, add_arrays_results)
write_results_to_file(
    [LIB_NAME, mode], BACKGROUND_CORRECTION, background_correction_results
)
