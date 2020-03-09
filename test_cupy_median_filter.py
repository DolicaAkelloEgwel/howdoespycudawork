import cupy as cp
import numpy as np
from cupy_test_utils import (
    CupyImplementation,
    LIB_NAME,
    free_memory_pool,
    REFLECT_MODE,
    cupy_median_filter,
)
from imagingtester import (
    USE_CUPY_NONPINNED_MEMORY,
    SIZES_SUBSET,
    DTYPE,
    N_RUNS,
    FILTER_SIZE,
)
from numpy_scipy_imaging_filters import scipy_median_filter
from write_and_read_results import ARRAY_SIZES, write_results_to_file

if USE_CUPY_NONPINNED_MEMORY:
    pinned_memory_mode = [True, False]
else:
    pinned_memory_mode = [True]

# These need to be odd values and need to be equal
filter_height = 3
filter_width = 3
filter_size = (filter_height, filter_width)

cp_data = cp.random.uniform(low=0, high=10, size=(5, 5, 5))
np_data = cp_data.get()

# Create a padded array in the GPU
pad_height = filter_height // 2
pad_width = filter_width // 2
padded_data = cp.pad(
    cp_data,
    pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width)),
    mode=REFLECT_MODE,
)

# Run the median filter on the GPU
cupy_median_filter(
    data=cp_data,
    padded_data=padded_data,
    filter_height=filter_height,
    filter_width=filter_width,
)
# Run the scipy median filter
scipy_median_filter(np_data, size=filter_size)
# Check that the results match
assert np.allclose(np_data, cp_data.get())

free_memory_pool([cp_data])

for use_pinned_memory in pinned_memory_mode:

    # Create empty lists for storing results
    median_filter_results = []

    if use_pinned_memory:
        memory_string = "with pinned memory"
    else:
        memory_string = "without pinned memory"

    for size in ARRAY_SIZES[:SIZES_SUBSET]:

        imaging_obj = CupyImplementation(size, DTYPE, use_pinned_memory)

        avg_mf = imaging_obj.timed_median_filter(N_RUNS, FILTER_SIZE)

        if avg_mf > 0:
            median_filter_results.append(avg_mf)

        write_results_to_file(
            [LIB_NAME, memory_string], "median filter", median_filter_results
        )
