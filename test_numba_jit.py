from numba import cuda
import time
import numpy as np

from imagingtester import (
    N_RUNS,
    DTYPE,
    create_arrays,
    SIZES_SUBSET,
    get_array_partition_indices,
    num_partitions_needed,
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
)
from numba_test_utils import get_free_bytes, NumbaImplementation
from numpy_scipy_imaging_filters import numpy_background_correction
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


# @cuda.jit
# def median_filter(data_array, padded_array, filter_height, filter_width):
#
#     n_images, img_x, img_y = cuda.grid(3)
#
#     if n_images >= data_array.shape[0] or img_x >= data_array.shape[1] or img_y >= data_array.shape[2]:
#         return
#
#     for i in range(img_x, img_x + filter_height):
#         for j in range(img_y, img_y + filter_width):
#


class NumbaCudaJitImplementation(NumbaImplementation):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)

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

    def get_time(self):
        self.synchronise()
        return time.time()

    def timed_imaging_operation(
        self, runs, alg, alg_name, n_arrs_needed, n_gpu_arrs_needed
    ):

        # Synchronize and free memory before making an assessment about available space
        self.clear_cuda_memory()

        n_partitions_needed = num_partitions_needed(
            self.cpu_arrays[0], n_gpu_arrs_needed, get_free_bytes()
        )

        transfer_time = 0
        operation_time = 0

        if n_partitions_needed == 1:

            cpu_result_array = np.empty_like(self.cpu_arrays[0])

            # Time transfer from CPU to GPU
            start = self.get_time()
            gpu_input_arrays = self._send_arrays_to_gpu(
                self.cpu_arrays[:n_arrs_needed], n_gpu_arrs_needed
            )
            gpu_output_array = self._send_arrays_to_gpu(
                [cpu_result_array], n_gpu_arrs_needed
            )[0]
            transfer_time += self.get_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += self.time_function(
                    lambda: alg(*gpu_input_arrays[:n_arrs_needed], gpu_output_array)
                )

            stream = cuda.stream()
            self.streams.append(stream)

            # Time the transfer from GPU to CPU
            transfer_time += self.time_function(
                lambda: gpu_output_array.copy_to_host(cpu_result_array, stream)
            )

            # Free the GPU arrays
            self.clear_cuda_memory(gpu_input_arrays)

        else:

            # Determine the number of partitions required again (to be on the safe side)
            n_partitions_needed = num_partitions_needed(
                self.cpu_arrays[0], n_gpu_arrs_needed, get_free_bytes()
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

                cpu_result_array = np.empty_like(split_cpu_arrays[i])

                # Time transferring the segments to the GPU
                start = self.get_time()
                gpu_input_arrays = self._send_arrays_to_gpu(
                    split_cpu_arrays, n_gpu_arrs_needed
                )
                gpu_output_array_list = self._send_arrays_to_gpu(
                    [cpu_result_array], n_gpu_arrs_needed
                )
                transfer_time += self.get_time() - start

                if not gpu_input_arrays:
                    return 0

                if not gpu_output_array_list:
                    return 0

                gpu_output_array = gpu_output_array_list[0]

                # Carry out the operation on the slices
                for _ in range(runs):
                    operation_time += self.time_function(
                        lambda: alg(*gpu_input_arrays[:n_arrs_needed], gpu_output_array)
                    )

                stream = cuda.stream()
                self.streams.append(stream)

                transfer_time += self.time_function(
                    lambda: gpu_output_array.copy_to_host(cpu_result_array, stream)
                )

                # Free GPU arrays and partition arrays
                self.clear_cuda_memory(gpu_input_arrays + [gpu_output_array])

        if transfer_time > 0 and operation_time > 0:
            self.print_operation_times(operation_time, alg_name, runs, transfer_time)

        self.synchronise()

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

    imaging_obj = NumbaCudaJitImplementation(size, DTYPE)

    avg_add = imaging_obj.timed_imaging_operation(
        N_RUNS, add_arrays_with_set_block_and_grid, "adding", 2, 3
    )
    avg_bc = imaging_obj.timed_imaging_operation(
        N_RUNS,
        background_correction_with_set_block_and_grid,
        "background correction",
        3,
        4,
    )

    if avg_add > 0:
        add_arrays_results.append(avg_add)
    if avg_bc > 0:
        background_correction_results.append(avg_bc)

write_results_to_file([LIB_NAME, mode], ADD_ARRAYS, add_arrays_results)
write_results_to_file(
    [LIB_NAME, mode], BACKGROUND_CORRECTION, background_correction_results
)
