import time

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
    USE_NONPINNED_MEMORY,
    memory_needed_for_arrays,
)
from imagingtester import num_partitions_needed as number_of_partitions_needed
from numpy_background_correction import numpy_background_correction
from write_and_read_results import (
    write_results_to_file,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
    ARRAY_SIZES,
)

if USE_NONPINNED_MEMORY:
    pinned_memory_mode = [True, False]
else:
    pinned_memory_mode = [True]

LIB_NAME = "cupy"
MAX_CUPY_MEMORY = 0.9  # Anything exceeding this seems to make malloc fail for me


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
    norm_divide = cp.subtract(flat, dark)
    norm_divide[norm_divide == 0] = MINIMUM_PIXEL_VALUE
    data = cp.subtract(data, dark)
    data = cp.true_divide(data, norm_divide)
    data = cp.clip(
        data, clip_min, clip_max
    )  # For some reason using the 'out' parameter doesn't work


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

    def _send_arrays_to_gpu_with_pinned_memory(self, cpu_arrays, n_gpu_arrays_needed):
        """
        Transfer the arrays to the GPU using pinned memory. Should make data transfer quicker.
        """
        gpu_arrays = []

        for i in range(len(cpu_arrays)):
            try:
                pinned_memory = _create_pinned_memory(cpu_arrays[i])
                gpu_array = cp.empty(pinned_memory.shape, dtype=self.dtype)
                array_stream = cp.cuda.Stream(non_blocking=True)
                gpu_array.set(pinned_memory, stream=array_stream)
                gpu_arrays.append(gpu_array)
            except cp.cuda.memory.OutOfMemoryError:
                self.print_memory_after_exception(
                    cpu_arrays, gpu_arrays, n_gpu_arrays_needed
                )
                return []

        return gpu_arrays

    def _send_arrays_to_gpu_without_pinned_memory(
        self, cpu_arrays, n_gpu_arrays_needed
    ):
        """
        Transfer the arrays to the GPU without using pinned memory.
        """
        gpu_arrays = []
        for cpu_array in cpu_arrays:
            try:
                gpu_array = cp.asarray(cpu_array)
            except cp.cuda.memory.OutOfMemoryError:
                self.print_memory_after_exception(
                    cpu_arrays, gpu_arrays, n_gpu_arrays_needed
                )
                return []
            gpu_arrays.append(gpu_array)

        gpu_arrays = [cp.asarray(cpu_array) for cpu_array in cpu_arrays]
        return gpu_arrays

    def print_memory_after_exception(self, cpu_arrays, gpu_arrays, n_gpu_arrays_needed):
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
            memory_needed_for_arrays(cpu_arrays[0], n_gpu_arrays_needed),
        )
        free_memory_pool(gpu_arrays)

    def timed_imaging_operation(
        self, runs, alg, alg_name, n_arrs_needed, n_gpu_arrs_needed
    ):

        # Synchronize and free memory before making an assessment about available space
        free_memory_pool()

        # Determine the number of partitions required
        n_partitions_needed = number_of_partitions_needed(
            self.cpu_arrays[0], n_gpu_arrs_needed, mempool.get_limit()
        )

        transfer_time = 0
        operation_time = 0

        if n_partitions_needed == 1:

            # Time the transfer from CPU to GPU
            start = get_synchronized_time()
            gpu_arrays = self._send_arrays_to_gpu(
                self.cpu_arrays[:n_arrs_needed], n_gpu_arrs_needed
            )
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
                self.cpu_arrays[0], n_gpu_arrs_needed, mempool.get_limit()
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
                gpu_arrays = self._send_arrays_to_gpu(
                    split_cpu_arrays, n_gpu_arrs_needed
                )
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


set_allocator(MemoryPool(malloc_managed).malloc)

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
    cp.random.uniform(low=0.0, high=20, size=(5, 5, 5)) for _ in range(3)
]
cp_data, cp_dark, cp_flat = random_test_arrays
np_data, np_dark, np_flat = [cp_arr.get() for cp_arr in random_test_arrays]
cupy_background_correction(cp_data, cp_dark, cp_flat)
numpy_background_correction(np_data, np_dark, np_flat)
assert np.allclose(np_data, cp_data.get())

# Getting rid of test arrays
free_memory_pool(random_test_arrays + [all_one])

for use_pinned_memory in pinned_memory_mode:

    # Create empty lists for storing results
    add_arrays_results = []
    background_correction_results = []

    for size in ARRAY_SIZES[:SIZES_SUBSET]:

        imaging_obj = CupyImplementation(size, DTYPE, use_pinned_memory)

        avg_add = imaging_obj.timed_imaging_operation(
            N_RUNS, cupy_add_arrays, "adding", 2, 2
        )
        avg_bc = imaging_obj.timed_imaging_operation(
            N_RUNS, cupy_background_correction, "background correction", 3, 5
        )

        if avg_add > 0:
            add_arrays_results.append(avg_add)
        if avg_bc > 0:
            background_correction_results.append(avg_bc)

    if use_pinned_memory:
        memory_string = "with pinned memory"
    else:
        memory_string = "without pinned memory"

    write_results_to_file([LIB_NAME, memory_string], ADD_ARRAYS, add_arrays_results)
    write_results_to_file(
        [LIB_NAME, memory_string], BACKGROUND_CORRECTION, background_correction_results
    )
