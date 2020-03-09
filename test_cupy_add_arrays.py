import cupy as cp
from cupy_test_utils import (
    CupyImplementation,
    cupy_add_arrays,
    LIB_NAME,
    free_memory_pool,
)
from imagingtester import USE_CUPY_NONPINNED_MEMORY, SIZES_SUBSET, DTYPE, N_RUNS
from write_and_read_results import ARRAY_SIZES, write_results_to_file, ADD_ARRAYS

if USE_CUPY_NONPINNED_MEMORY:
    pinned_memory_mode = [True, False]
else:
    pinned_memory_mode = [True]

# Checking that cupy will change the value of the array
all_one = cp.ones((1, 1, 1))
cupy_add_arrays(all_one, all_one)
assert cp.all(all_one == 2)

free_memory_pool([all_one])

for use_pinned_memory in pinned_memory_mode:

    # Create empty lists for storing results
    add_arrays_results = []

    if use_pinned_memory:
        memory_string = "with pinned memory"
    else:
        memory_string = "without pinned memory"

    for size in ARRAY_SIZES[:SIZES_SUBSET]:

        imaging_obj = CupyImplementation(size, DTYPE, use_pinned_memory)

        avg_add = imaging_obj.timed_imaging_operation(
            N_RUNS, cupy_add_arrays, "adding", 2
        )

        if avg_add > 0:
            add_arrays_results.append(avg_add)

        write_results_to_file([LIB_NAME, memory_string], ADD_ARRAYS, add_arrays_results)
