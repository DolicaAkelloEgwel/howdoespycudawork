import cupy as cp
import numpy as np
from cupy_test_utils import (
    CupyImplementation,
    LIB_NAME,
    free_memory_pool,
    cupy_background_correction,
)
from imagingtester import USE_CUPY_NONPINNED_MEMORY, SIZES_SUBSET, DTYPE, N_RUNS
from numpy_scipy_imaging_filters import numpy_background_correction
from write_and_read_results import (
    ARRAY_SIZES,
    write_results_to_file,
    BACKGROUND_CORRECTION,
)

if USE_CUPY_NONPINNED_MEMORY:
    pinned_memory_mode = [True, False]
else:
    pinned_memory_mode = [True]

# Checking the two background corrections get the same result
random_test_arrays = [
    cp.random.uniform(low=0.0, high=20, size=(10, 10, 10)) for _ in range(3)
]
cp_data, cp_dark, cp_flat = random_test_arrays
np_data, np_dark, np_flat = [cp_arr.get() for cp_arr in random_test_arrays]
cupy_background_correction(cp_data, cp_dark, cp_flat)
numpy_background_correction(np_data, np_dark, np_flat)
assert np.allclose(np_data, cp_data.get())

free_memory_pool(random_test_arrays)

for use_pinned_memory in pinned_memory_mode:

    # Create empty lists for storing results
    background_correction_results = []

    if use_pinned_memory:
        memory_string = "with pinned memory"
    else:
        memory_string = "without pinned memory"

    for size in ARRAY_SIZES[:SIZES_SUBSET]:

        imaging_obj = CupyImplementation(size, DTYPE, use_pinned_memory)

        avg_bc = imaging_obj.timed_imaging_operation(
            N_RUNS, cupy_background_correction, "background correction", 3
        )

        if avg_bc > 0:
            background_correction_results.append(avg_bc)

        write_results_to_file(
            [LIB_NAME, memory_string],
            BACKGROUND_CORRECTION,
            background_correction_results,
        )
