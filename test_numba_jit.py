from numba import cuda
import time
import numpy as np

from imagingtester import (
    ImagingTester,
    N_RUNS,
    DTYPE,
    create_arrays,
    SIZES_SUBSET,
    partition_arrays,
    PRINT_INFO,
    num_partitions_needed,
    memory_needed_for_arrays,
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
)
from numba_test_utils import STREAM, get_free_bytes, get_used_bytes, get_total_bytes
from numpy_background_correction import numpy_background_correction
from write_and_read_results import (
    write_results_to_file,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
    ARRAY_SIZES,
)

LIB_NAME = "numba"
mode = "jit"

TPB = cuda.get_current_device().WARP_SIZE
GRIDDIM = (TPB // 4, TPB // 2, TPB)
BLOCKDIM = (TPB // 16, TPB // 8, TPB)


@cuda.jit
def add_arrays(arr1, arr2, out):
    i, j, k = cuda.grid(3)

    if i < arr1.shape[0] and j < arr1.shape[1] and k < arr1.shape[2]:
        out[i, j, k] = arr1[i][j][k] + arr2[i][j][k]


@cuda.jit
def background_correction(dark, data, flat, out, clip_min, clip_max):
    i, j, k = cuda.grid(3)

    if i >= data.shape[0] or j >= data.shape[1] or k >= data.shape[2]:
        return

    norm_divide = flat[i, j, k] - dark[i, j, k]

    if norm_divide == 0:
        norm_divide = MINIMUM_PIXEL_VALUE

    out[i, j, k] = (data[i, j, k] - dark[i, j, k]) / norm_divide

    if out[i, j, k] < clip_min:
        out[i, j, k] = clip_min
    if out[i, j, k] > clip_max:
        out[i, j, k] = clip_max


class NumbaImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.warm_up()
        self.lib_name = LIB_NAME

    def warm_up(self):
        """
        Give the CUDA functions a chance to compile.
        """
        warm_up_arrays = create_arrays((1, 1, 1), self.dtype)
        add_arrays(*warm_up_arrays)
        background_correction(
            *warm_up_arrays,
            np.empty_like(warm_up_arrays[0]),
            MINIMUM_PIXEL_VALUE,
            MAXIMUM_PIXEL_VALUE
        )

    def get_synchronized_time(self):
        STREAM.synchronize()
        return time.time()

    def time_function(self, func):
        start = self.get_synchronized_time()
        func()
        return self.get_synchronized_time() - start

    def clear_cuda_memory(self, split_arrays=[]):

        cuda.synchronize()
        STREAM.synchronize()

        if PRINT_INFO:
            print("Free bytes before clearing memory:", get_free_bytes())

        if split_arrays:
            for array in split_arrays:
                del array
                array = None
        cuda.current_context().deallocations.clear()
        STREAM.synchronize()

        if PRINT_INFO:
            print("Free bytes after clearing memory:", get_free_bytes())

    def _send_arrays_to_gpu(self, arrays_to_transfer):

        gpu_arrays = []

        with cuda.pinned(*arrays_to_transfer):
            for arr in arrays_to_transfer:
                gpu_arrays.append(cuda.to_device(arr, STREAM))
        return gpu_arrays

    def timed_imaging_operation(self, runs, alg, alg_name, n_arrs_needed):

        # Synchronize and free memory before making an assessment about available space
        self.clear_cuda_memory()

        n_partitions_needed = num_partitions_needed(
            self.cpu_arrays[:n_arrs_needed] + [np.empty_like(self.cpu_arrays[0])],
            get_free_bytes(),
        )

        transfer_time = 0
        operation_time = 0

        if n_partitions_needed == 1:

            cpu_result_array = np.empty_like(self.cpu_arrays[0])

            # Time transfer from CPU to GPU
            start = self.get_synchronized_time()
            gpu_input_arrays = self._send_arrays_to_gpu(self.cpu_arrays[:n_arrs_needed])
            gpu_output_array = self._send_arrays_to_gpu([cpu_result_array])[0]
            transfer_time += self.get_synchronized_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += self.time_function(
                    lambda: alg(*gpu_input_arrays[:n_arrs_needed], gpu_output_array)
                )

            # Time the transfer from GPU to CPU
            transfer_time += self.time_function(
                lambda: gpu_output_array.copy_to_host(cpu_result_array, STREAM)
            )

            # Free the GPU arrays
            self.clear_cuda_memory(gpu_input_arrays + [gpu_output_array])

        else:

            print("Partitions:", n_partitions_needed)

            # Split the arrays
            split_arrays = partition_arrays(
                self.cpu_arrays[:n_arrs_needed], n_partitions_needed
            )

            for i in range(n_partitions_needed):

                # Retrieve the segments used for this iteration of the operation
                split_cpu_arrays = [
                    split_arrays[k][i] for k in range(len(split_arrays))
                ]

                cpu_result_array = np.empty_like(split_cpu_arrays[i])

                try:

                    # Time transferring the segments to the GPU
                    start = self.get_synchronized_time()
                    gpu_input_arrays = self._send_arrays_to_gpu(split_cpu_arrays)
                    gpu_output_array = self._send_arrays_to_gpu([cpu_result_array])[0]
                    transfer_time += self.get_synchronized_time() - start

                except cuda.cudadrv.driver.CudaAPIError:

                    # This shouldn't happen provided partitioning is working correctly...
                    print(
                        "Failed to make %s GPU arrays of size %s."
                        % (n_arrs_needed + 1, split_cpu_arrays[0].shape)
                    )
                    print(
                        "Used bytes:",
                        get_used_bytes(),
                        "/ Total bytes:",
                        get_total_bytes(),
                        "/ Space needed:",
                        memory_needed_for_arrays(split_cpu_arrays + [cpu_result_array]),
                    )
                    break

                # Carry out the operation on the slices
                for _ in range(runs):
                    operation_time += self.time_function(
                        lambda: alg(*gpu_input_arrays[:n_arrs_needed], gpu_output_array)
                    )

                transfer_time += self.time_function(
                    lambda: gpu_output_array.copy_to_host(cpu_result_array, STREAM)
                )

                # Free GPU arrays and partition arrays
                self.clear_cuda_memory(
                    split_cpu_arrays + gpu_input_arrays + [gpu_output_array]
                )

        if transfer_time > 0 and operation_time > 0:
            self.print_operation_times(operation_time, alg_name, runs, transfer_time)

        STREAM.synchronize()

        return transfer_time + operation_time / runs


practice_array = np.ones(shape=(5, 5, 5)).astype(DTYPE)
jit_result = np.empty_like(practice_array).astype(DTYPE)
add_arrays[GRIDDIM, BLOCKDIM](practice_array, practice_array, jit_result)
assert np.all(jit_result == 2)

# # Checking the two background corrections get the same result
np_data, np_dark, np_flat = [
    np.random.uniform(low=0.0, high=20, size=(5, 5, 5)) for _ in range(3)
]
jit_result = np.empty_like(np_data).astype(DTYPE)
background_correction[GRIDDIM, BLOCKDIM](
    np_dark, np_data, np_flat, jit_result, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
)
numpy_background_correction(
    np_dark, np_data, np_flat, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
)
assert np.allclose(np_data, jit_result)


def add_arrays_with_set_block_and_grid(arr1, arr2, out):
    add_arrays[GRIDDIM, BLOCKDIM](arr1, arr2, out)


def background_correction_with_set_block_and_grid(dark, data, flat, out):
    background_correction[GRIDDIM, BLOCKDIM](
        dark, data, flat, out, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
    )


add_arrays_results = []
background_correction_results = []

for size in ARRAY_SIZES[:SIZES_SUBSET]:

    imaging_obj = NumbaImplementation(size, DTYPE)

    try:
        avg_add = imaging_obj.timed_imaging_operation(
            N_RUNS, add_arrays_with_set_block_and_grid, "adding", 2
        )
        avg_bc = imaging_obj.timed_imaging_operation(
            N_RUNS,
            background_correction_with_set_block_and_grid,
            "background correction",
            3,
        )

        add_arrays_results.append(avg_add)
        background_correction_results.append(avg_bc)

    except cuda.cudadrv.driver.CudaAPIError:
        if PRINT_INFO:
            print("Can't operate on arrays with size:", size)
            print("Free bytes during CUDA error:", get_free_bytes())
        imaging_obj.clear_cuda_memory()
        break

write_results_to_file([LIB_NAME, mode], ADD_ARRAYS, add_arrays_results)
write_results_to_file(
    [LIB_NAME, mode], BACKGROUND_CORRECTION, background_correction_results
)
