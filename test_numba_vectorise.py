from numba import vectorize, cuda
import time
import numpy as np

from imagingtester import (
    ImagingTester,
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
    N_RUNS,
    DTYPE,
    create_arrays,
    SIZES_SUBSET,
    partition_arrays,
    PRINT_INFO,
    num_partitions_needed,
    memory_needed_for_arrays,
)
from numpy_background_correction import numpy_background_correction
from write_and_read_results import (
    write_results_to_file,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
    ARRAY_SIZES,
)

LIB_NAME = "numba"
mode = "vectorise"


def get_free_bytes():
    return cuda.current_context().get_memory_info()[0]


def get_used_bytes():
    return cuda.current_context().get_memory_info()[1] - get_free_bytes()


@vectorize(["{0}({0},{0})".format(DTYPE)], target="cuda")
def add_arrays(elem1, elem2):
    return elem1 + elem2


@vectorize("{0}({0},{0},{0},{0},{0})".format(DTYPE), target="cuda")
def background_correction(data, dark, flat, clip_min, clip_max):

    norm_divide = flat - dark

    if norm_divide == 0:
        norm_divide = MINIMUM_PIXEL_VALUE

    data -= dark
    data /= norm_divide

    if data < clip_min:
        data = clip_min
    if data > clip_max:
        data = clip_max

    return data


# @cuda.jit("void({0}[:,:,:],{0}[:,:,:],{0}[:,:,:])".format(DTYPE))
# @cuda.jit
# def cuda_jit_add_arrays(arr1, arr2, out):
#
#     i, j, k = cuda.grid(3)
#
#     if i < arr1.shape[0] and j < arr1.shape[1] and k < arr1.shape[2]:
#         out[i, j, k] = arr1[i, j, k] + arr2[i, j, k]
#
#
# @cuda.jit
# def cuda_jit_background_correction(
#     data, dark, flat, clip_min=MINIMUM_PIXEL_VALUE, clip_max=MAXIMUM_PIXEL_VALUE
# ):
#     i, j, k = cuda.grid(3)
#
#     if i < data.shape[0] and j < data.shape[1] and k < data.shape[2]:
#         data[i][j][k] -= dark[i][j][k]
#         flat[i][j][k] -= dark[i][j][k]
#
#         if flat[i][j][k] > 0:
#             data[i][j][k] /= flat[i][j][k]
#         else:
#             data[i][j][k] /= MINIMUM_PIXEL_VALUE
#
#         if data[i][j][k] < clip_min:
#             data[i][j][k] = clip_min
#         elif data[i][j][k] > clip_max:
#             data[i][j][k] = clip_max


stream = cuda.stream()


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
        add_arrays(*warm_up_arrays[:2])
        background_correction(*warm_up_arrays, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE)

    def get_synchronized_time(self):
        stream.synchronize()
        return time.time()

    def time_function(self, func):
        start = self.get_synchronized_time()
        func()
        return self.get_synchronized_time() - start

    def clear_cuda_memory(self, split_arrays=[]):

        cuda.synchronize()
        stream.synchronize()

        if PRINT_INFO:
            print("Free bytes before clearing memory:", get_free_bytes())

        if split_arrays:
            for array in split_arrays:
                del array
                array = None
        cuda.current_context().deallocations.clear()
        stream.synchronize()

        if PRINT_INFO:
            print("Free bytes after clearing memory:", get_free_bytes())

    def _send_arrays_to_gpu(self, cpu_arrays):

        gpu_arrays = []
        arrays_to_transfer = cpu_arrays

        with cuda.pinned(*arrays_to_transfer):
            for arr in arrays_to_transfer:
                gpu_arrays.append(cuda.to_device(arr, stream))

        return gpu_arrays

    def timed_imaging_operation(self, runs, alg, alg_name, n_arrs_needed):

        # Synchronize and free memory before making an assessment about available space
        self.clear_cuda_memory()

        n_partitions_needed = num_partitions_needed(
            self.cpu_arrays[:n_arrs_needed], get_free_bytes()
        )

        transfer_time = 0
        operation_time = 0

        if n_partitions_needed == 1:

            # Time transfer from CPU to GPU
            start = self.get_synchronized_time()
            gpu_arrays = self._send_arrays_to_gpu(self.cpu_arrays[:n_arrs_needed])
            transfer_time += self.get_synchronized_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += self.time_function(
                    lambda: alg(*gpu_arrays[:n_arrs_needed])
                )

            # # Time the transfer from GPU to CPU
            # transfer_time += self.time_function(
            #     lambda: gpu_result_array.copy_to_host(cpu_result_array, stream)
            # )

            # Free the GPU arrays
            self.clear_cuda_memory(gpu_arrays)

        else:

            # Split the arrays
            split_arrays = partition_arrays(
                self.cpu_arrays[:n_arrs_needed], n_partitions_needed
            )

            for i in range(n_partitions_needed):

                # Retrieve the segments used for this iteration of the operation
                split_cpu_arrays = [
                    split_arrays[k][i] for k in range(len(split_arrays))
                ]

                try:

                    # Time transferring the segments to the GPU
                    start = self.get_synchronized_time()
                    gpu_arrays = self._send_arrays_to_gpu(split_cpu_arrays)
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
                        get_free_bytes(),
                        "/ Space needed:",
                        memory_needed_for_arrays(split_cpu_arrays),
                    )
                    break

                # Carry out the operation on the slices
                for _ in range(runs):
                    operation_time += self.time_function(
                        lambda: alg(*gpu_arrays[:n_arrs_needed])
                    )

                # transfer_time += self.time_function(
                #     lambda: gpu_result_array.copy_to_host(cpu_result_array, stream)
                # )

                # Free GPU arrays and partition arrays
                self.clear_cuda_memory(
                    split_cpu_arrays + [gpu_arrays, gpu_result_array]
                )

        if transfer_time > 0 and operation_time > 0:
            self.print_operation_times(operation_time, alg_name, runs, transfer_time)

        return transfer_time + operation_time / runs


practice_array = np.ones(shape=(5, 5, 5)).astype(DTYPE)
jit_result = np.empty_like(practice_array).astype(DTYPE)
vect_result = add_arrays(practice_array, practice_array)
assert np.all(vect_result == 2)

add_arrays_results = []
background_correction_results = []


def background_correction_fixed_clip(dark, data, flat):
    return background_correction(
        data, dark, flat, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
    )


for size in ARRAY_SIZES[:SIZES_SUBSET]:

    imaging_obj = NumbaImplementation(size, DTYPE)

    try:
        avg_add = imaging_obj.timed_imaging_operation(N_RUNS, add_arrays, "adding", 2)
        avg_bc = imaging_obj.timed_imaging_operation(
            N_RUNS, background_correction_fixed_clip, "background correction", 3
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
