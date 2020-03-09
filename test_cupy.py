import time
from typing import List

import cupy as cp
import numpy as np
from cupy.cuda.memory import set_allocator, MemoryPool, malloc_managed

from imagingtester import (
    ImagingTester,
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
    create_arrays,
    N_RUNS,
    SIZES_SUBSET,
    DTYPE,
    PRINT_INFO,
    get_array_partition_indices,
    USE_CUPY_NONPINNED_MEMORY,
    memory_needed_for_arrays,
    load_median_filter_file,
    FILTER_SIZE,
)
from imagingtester import num_partitions_needed as number_of_partitions_needed
from numpy_scipy_imaging_filters import numpy_background_correction, scipy_median_filter
from write_and_read_results import (
    write_results_to_file,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
    ARRAY_SIZES,
)

if USE_CUPY_NONPINNED_MEMORY:
    pinned_memory_mode = [True, False]
else:
    pinned_memory_mode = [True]

LIB_NAME = "cupy"
MAX_CUPY_MEMORY = 0.9  # Anything exceeding this seems to make malloc fail for me

REFLECT_MODE = "reflect"


def print_memory_metrics():
    """
    Print some information about how much space is being used on the GPU.
    """
    if not PRINT_INFO:
        return
    print("Used bytes:", mempool.used_bytes(), "/ Total bytes:", mempool.total_bytes())


def synchronise():
    cp.cuda.Stream.null.synchronize()
    cp.cuda.runtime.deviceSynchronize()


def get_synchronized_time():
    """
    Get the time after calling the cuda synchronize method. This should ensure the GPU has completed whatever it was
    doing before getting the time.
    """
    synchronise()
    return time.time()


def free_memory_pool(arrays=[]):
    """
    Delete the existing GPU arrays and free blocks so that successive calls to `_send_arrays_to_gpu` don't lead to any
    problems.
    """
    synchronise()
    if arrays:
        for arr in arrays:
            del arr
            arr = None
        del arrays
    synchronise()
    mempool.free_all_blocks()


def _create_pinned_memory(cpu_array):
    """
    Use pinned memory as opposed to `asarray`. This allegedly this makes transferring quicker.
    :param cpu_array: The numpy array.
    :return: src
    """
    mem = cp.cuda.alloc_pinned_memory(cpu_array.nbytes)
    src = np.frombuffer(mem, cpu_array.dtype, cpu_array.size).reshape(cpu_array.shape)
    src[...] = cpu_array
    return src


def time_function(func):
    """
    Time an operation using a call to cupy's deviceSynchronize.
    :param func: The function to be timed.
    :return: The time the function took to complete its execution in seconds.
    """
    start = get_synchronized_time()
    func()
    return get_synchronized_time() - start


def get_free_bytes():
    free_bytes = mempool.free_bytes()
    if free_bytes > 0:
        return free_bytes
    return mempool.get_limit()


def cupy_add_arrays(arr1, arr2):
    """
    Add two arrays. Guaranteed to be slower on the GPU as it's a simple operation.
    """
    arr1 += arr2


def double_array(arr1):
    """
    Double an array. Simply used to check that cupy works as expected.
    """
    arr1 *= 2


def cupy_background_correction(
    data, dark, flat, clip_min=MINIMUM_PIXEL_VALUE, clip_max=MAXIMUM_PIXEL_VALUE
):
    """
    Carry out something akin to background correction in Mantid Imaging using cupy.
    :param data: The fake data array.
    :param dark: The fake dark array.
    :param flat: The fake flat array.
    :param clip_min: Minimum clipping value.
    :param clip_max: Maximum clipping value.
    """
    flat = cp.subtract(flat, dark)
    flat[flat == 0] = MINIMUM_PIXEL_VALUE
    data = cp.subtract(data, dark)
    data = cp.true_divide(data, flat)
    data = cp.clip(
        data, clip_min, clip_max
    )  # For some reason using the 'out' parameter doesn't work


loaded_from_source = load_median_filter_file()

median_filter_module = cp.RawModule(code=loaded_from_source, backend="nvcc")
median_filter = median_filter_module.get_function("median_filter")


def cupy_median_filter(data, padded_data, filter_height, filter_width):
    N = 10
    median_filter(
        (N, N, N),
        (N, N, N),
        (
            data,
            padded_data,
            data.shape[0],
            data.shape[1],
            data.shape[2],
            filter_height,
            filter_width,
        ),
    )


class CupyImplementation(ImagingTester):
    def __init__(self, size, dtype, pinned_memory=False):
        super().__init__(size, dtype)

        # Determine how to pass data to the GPU based on the pinned_memory argument
        if pinned_memory:
            self._send_arrays_to_gpu = self._send_arrays_to_gpu_with_pinned_memory
        else:
            self._send_arrays_to_gpu = self._send_arrays_to_gpu_without_pinned_memory

        self.warm_up()
        self.lib_name = LIB_NAME

    def warm_up(self):
        """
        Give CUDA an opportunity to compile these functions?
        """
        warm_up_arrays = [
            cp.asarray(cpu_array) for cpu_array in create_arrays((1, 1, 1), self.dtype)
        ]
        cupy_background_correction(*warm_up_arrays)
        cupy_add_arrays(*warm_up_arrays[:2])

    def _send_arrays_to_gpu_with_pinned_memory(self, cpu_arrays):
        """
        Transfer the arrays to the GPU using pinned memory. Should make data transfer quicker.
        """
        gpu_arrays = []

        if not isinstance(cpu_arrays, List):
            cpu_arrays = [cpu_arrays]

        for i in range(len(cpu_arrays)):
            try:
                pinned_memory = _create_pinned_memory(cpu_arrays[i])
                gpu_array = cp.empty(pinned_memory.shape, dtype=self.dtype)
                array_stream = cp.cuda.Stream(non_blocking=True)
                gpu_array.set(pinned_memory, stream=array_stream)
                gpu_arrays.append(gpu_array)
            except cp.cuda.memory.OutOfMemoryError:
                self.print_memory_after_exception(cpu_arrays, gpu_arrays)
                return []

        if len(gpu_arrays) == 1:
            return gpu_arrays[0]
        return gpu_arrays

    def _send_arrays_to_gpu_without_pinned_memory(self, cpu_arrays):
        """
        Transfer the arrays to the GPU without using pinned memory.
        """
        gpu_arrays = []

        if not isinstance(cpu_arrays, List):
            cpu_arrays = [cpu_arrays]

        for cpu_array in cpu_arrays:
            try:
                gpu_array = cp.asarray(cpu_array)
            except cp.cuda.memory.OutOfMemoryError:
                self.print_memory_after_exception(cpu_arrays, gpu_arrays)
                return []
            gpu_arrays.append(gpu_array)

        gpu_arrays = [cp.asarray(cpu_array) for cpu_array in cpu_arrays]

        if len(gpu_arrays) == 1:
            return gpu_arrays[0]
        return gpu_arrays

    def print_memory_after_exception(self, cpu_arrays, gpu_arrays):
        print(
            "Failed to make %s GPU arrays of size %s."
            % (len(cpu_arrays), cpu_arrays[0].shape)
        )
        print(
            "Used bytes:",
            mempool.used_bytes(),
            "/ Free bytes:",
            mempool.free_bytes(),
            "/ Space needed:",
            memory_needed_for_arrays(cpu_arrays),
        )
        free_memory_pool(gpu_arrays)

    def timed_imaging_operation(self, runs, alg, alg_name, n_arrs_needed):

        # Synchronize and free memory before making an assessment about available space
        free_memory_pool()

        # Determine the number of partitions required
        n_partitions_needed = number_of_partitions_needed(
            self.cpu_arrays, get_free_bytes()
        )

        transfer_time = 0
        operation_time = 0

        if n_partitions_needed == 1:

            # Time the transfer from CPU to GPU
            start = get_synchronized_time()
            gpu_arrays = self._send_arrays_to_gpu(self.cpu_arrays[:n_arrs_needed])
            transfer_time = get_synchronized_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += time_function(
                    lambda: alg(*gpu_arrays[:n_arrs_needed])
                )

            # Time the transfer from GPU to CPU
            transfer_time += time_function(gpu_arrays[0].get)

            # Free the GPU arrays
            free_memory_pool(gpu_arrays)

        else:

            # Determine the number of partitions required again (to be on the safe side)
            n_partitions_needed = number_of_partitions_needed(
                self.cpu_arrays[0], get_free_bytes()
            )

            indices = get_array_partition_indices(
                self.cpu_arrays[0].shape[0], n_partitions_needed
            )

            for i in range(n_partitions_needed):

                # Retrieve the segments used for this iteration of the operation
                split_cpu_arrays = [
                    cpu_array[indices[i][0] : indices[i][1] :, :]
                    for cpu_array in self.cpu_arrays
                ]

                # Time transferring the segments to the GPU
                start = get_synchronized_time()
                gpu_arrays = self._send_arrays_to_gpu(split_cpu_arrays)
                transfer_time += get_synchronized_time() - start

                # Return 0 when GPU is out of space
                if not gpu_arrays:
                    return 0

                try:
                    # Carry out the operation on the slices
                    for _ in range(runs):
                        operation_time += time_function(
                            lambda: alg(*gpu_arrays[:n_arrs_needed])
                        )
                except cp.cuda.memory.OutOfMemoryError as e:
                    print(
                        "Unable to make extra arrays during operation despite successful transfer."
                    )
                    print(e)
                    free_memory_pool(gpu_arrays)
                    return 0

                # Store time taken to transfer result
                transfer_time += time_function(gpu_arrays[0].get)

                # Free GPU arrays
                free_memory_pool(gpu_arrays)

        self.print_operation_times(
            operation_time=operation_time,
            operation_name=alg_name,
            runs=runs,
            transfer_time=transfer_time,
        )

        return transfer_time + operation_time / runs

    def timed_median_filter(self, runs, filter_size):

        # Synchronize and free memory before making an assessment about available space
        free_memory_pool()

        # Determine the number of partitions required (not taking the padding into account)
        n_partitions_needed = number_of_partitions_needed(
            self.cpu_arrays[:1], get_free_bytes()
        )

        transfer_time = 0
        operation_time = 0

        pad_height = filter_size[1] // 2
        pad_width = filter_size[0] // 2

        filter_height = filter_size[0]
        filter_width = filter_size[1]

        if n_partitions_needed == 1:

            # Time the transfer from CPU to GPU
            start = get_synchronized_time()
            gpu_data_array = self._send_arrays_to_gpu([self.cpu_arrays[0]])
            padded_array = cp.pad(
                gpu_data_array,
                pad_width=((0, 0), (pad_width, pad_width), (pad_height, pad_height)),
            )
            transfer_time = get_synchronized_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += time_function(
                    lambda: cupy_median_filter(
                        gpu_data_array, padded_data, filter_height, filter_width
                    )
                )

            # Time the transfer from GPU to CPU
            transfer_time += time_function(gpu_data_array[0].get)

            # Free the GPU arrays
            free_memory_pool([gpu_data_array, padded_array])

        else:

            indices = get_array_partition_indices(
                self.cpu_arrays[0].shape[0], n_partitions_needed
            )

            space_assessment_data_array = np.empty_like(
                self.cpu_arrays[0][indices[0][0] : indices[0][1] :, :]
            )
            space_assessment_padded_array = np.pad(
                space_assessment_data_array,
                pad_width=((0, 0), (pad_width, pad_width), (pad_height, pad_height)),
            )

            # Determine the number of partitions required again (to be on the safe side)
            n_partitions_needed = number_of_partitions_needed(
                [space_assessment_data_array, space_assessment_padded_array],
                get_free_bytes(),
            )

            gpu_arrays = self._send_arrays_to_gpu(
                [space_assessment_data_array, space_assessment_padded_array]
            )

            # Return 0 when GPU is out of space
            if not gpu_arrays:
                return 0

            gpu_data_array, gpu_padded_array = gpu_arrays

            for i in range(n_partitions_needed):

                # Retrieve the segments used for this iteration of the operation
                split_cpu_array = self.cpu_arrays[0][indices[i][0] : indices[i][1] :, :]

                # Time transferring the segments to the GPU
                start = get_synchronized_time()

                if split_cpu_array.shape == gpu_data_array.shape:
                    gpu_data_array.set(
                        split_cpu_array, cp.cuda.Stream(non_blocking=True)
                    )
                    gpu_padded_array.set(
                        np.pad(
                            split_cpu_array,
                            pad_width=(
                                (0, 0),
                                (pad_width, pad_width),
                                (pad_height, pad_height),
                            ),
                        ),
                        cp.cuda.Stream(non_blocking=True),
                    )
                else:

                    diff = gpu_data_array.shape[0] - split_cpu_array.shape[0]

                    expanded_cpu_array = np.pad(
                        split_cpu_array, pad_width=((0, diff), (0, 0), (0, 0))
                    )
                    gpu_data_array.set(
                        expanded_cpu_array, cp.cuda.Stream(non_blocking=True)
                    )

                    padded_cpu_array = np.pad(
                        expanded_cpu_array,
                        pad_width=(
                            (0, 0),
                            (pad_width, pad_width),
                            (pad_height, pad_height),
                        ),
                    )
                    gpu_padded_array.set(
                        padded_cpu_array, cp.cuda.Stream(non_blocking=True)
                    )

                transfer_time += get_synchronized_time() - start

                try:
                    # Carry out the operation on the slices
                    for _ in range(runs):
                        operation_time += time_function(
                            lambda: cupy_median_filter(
                                gpu_data_array,
                                gpu_padded_array,
                                filter_height,
                                filter_width,
                            )
                        )
                except cp.cuda.memory.OutOfMemoryError as e:
                    print(
                        "Unable to make extra arrays during operation despite successful transfer."
                    )
                    print(e)
                    free_memory_pool([gpu_data_array, gpu_padded_array])
                    return 0

                # Store time taken to transfer result
                transfer_time += time_function(gpu_data_array[0].get)

                # Free GPU arrays
                free_memory_pool([gpu_padded_array, gpu_data_array])

        self.print_operation_times(
            operation_time=operation_time,
            operation_name="Median Filter",
            runs=runs,
            transfer_time=transfer_time,
        )

        return transfer_time + operation_time / runs


# set_allocator(MemoryPool(malloc_managed).malloc)

# Allocate CUDA memory
mempool = cp.get_default_memory_pool()
with cp.cuda.Device(0):
    mempool.set_limit(fraction=MAX_CUPY_MEMORY)
mempool.malloc(mempool.get_limit())

# Checking that cupy will change the value of the array
all_one = cp.ones((1, 1, 1))
cupy_add_arrays(all_one, all_one)
assert cp.all(all_one == 2)

# Checking the two background corrections get the same result
random_test_arrays = [
    cp.random.uniform(low=0.0, high=20, size=(10, 10, 10)) for _ in range(3)
]
cp_data, cp_dark, cp_flat = random_test_arrays
np_data, np_dark, np_flat = [cp_arr.get() for cp_arr in random_test_arrays]
cupy_background_correction(cp_data, cp_dark, cp_flat)
numpy_background_correction(np_data, np_dark, np_flat)
assert np.allclose(np_data, cp_data.get())

# These need to be odd values and need to be equal
filter_height = 3
filter_width = 3
filter_size = (filter_height, filter_width)

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

# Getting rid of test arrays
free_memory_pool(random_test_arrays + [all_one])

for use_pinned_memory in pinned_memory_mode:

    # Create empty lists for storing results
    add_arrays_results = []
    background_correction_results = []
    median_filter_results = []

    if use_pinned_memory:
        memory_string = "with pinned memory"
    else:
        memory_string = "without pinned memory"

    for size in ARRAY_SIZES[:SIZES_SUBSET]:

        imaging_obj = CupyImplementation(size, DTYPE, use_pinned_memory)

        avg_med = imaging_obj.timed_median_filter(N_RUNS, FILTER_SIZE)
        avg_add = imaging_obj.timed_imaging_operation(
            N_RUNS, cupy_add_arrays, "adding", 2
        )
        avg_bc = imaging_obj.timed_imaging_operation(
            N_RUNS, cupy_background_correction, "background correction", 3
        )

        if avg_add > 0:
            add_arrays_results.append(avg_add)
        if avg_bc > 0:
            background_correction_results.append(avg_bc)
        if avg_med > 0:
            median_filter_results.append(avg_med)

        write_results_to_file([LIB_NAME, memory_string], ADD_ARRAYS, add_arrays_results)
        write_results_to_file(
            [LIB_NAME, memory_string],
            BACKGROUND_CORRECTION,
            background_correction_results,
        )
        write_results_to_file(
            [LIB_NAME, memory_string], "median filter", median_filter_results
        )
