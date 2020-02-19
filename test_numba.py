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
    TEST_PARALLEL_NUMBA,
    partition_arrays,
    NO_PRINT,
    num_partitions_needed,
    memory_needed_for_arrays,
)
from write_and_read_results import (
    write_results_to_file,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
    ARRAY_SIZES,
)

LIB_NAME = "numba"

PARALLEL_VECTORISE_MODE = "parallel vectorise"
CUDA_VECTORISE_MODE = "cuda vectorise"
CUDA_JIT_MODE = "cuda jit"

if TEST_PARALLEL_NUMBA:
    MODES = [PARALLEL_VECTORISE_MODE, CUDA_VECTORISE_MODE, CUDA_JIT_MODE]
else:
    MODES = [CUDA_VECTORISE_MODE, CUDA_JIT_MODE]


def get_free_bytes():
    meminfo = cuda.current_context().get_memory_info()
    return meminfo[0]


def get_used_bytes():
    meminfo = cuda.current_context().get_memory_info()
    return meminfo[1]


@vectorize(["{0}({0},{0})".format(DTYPE)], target="cuda")
def cuda_vectorise_add_arrays(elem1, elem2):
    return elem1 + elem2


@vectorize("{0}({0},{0},{0},int32,int32)".format(DTYPE), target="cuda")
def cuda_vectorise_background_correction(data, dark, flat, clip_min, clip_max):
    data -= dark
    flat -= dark
    if flat > 0:
        data /= flat
    else:
        data /= MINIMUM_PIXEL_VALUE

    if data < MINIMUM_PIXEL_VALUE:
        return MINIMUM_PIXEL_VALUE
    if data > MAXIMUM_PIXEL_VALUE:
        return MAXIMUM_PIXEL_VALUE
    return data


@cuda.jit
def cuda_jit_add_arrays(arr1, arr2):
    pos_x, pos_y, pos_z = cuda.grid(3)

    if pos_x < arr1.shape[0] and pos_y < arr1.shape[1] and pos_z < arr1.shape[2]:
        arr1[pos_x][pos_y][pos_z] += arr2[pos_x][pos_y][pos_z]


@cuda.jit
def cuda_jit_background_correction(
    data, dark, flat, clip_min=MINIMUM_PIXEL_VALUE, clip_max=MAXIMUM_PIXEL_VALUE
):
    i, j, k = cuda.grid(3)

    if i < data.shape[0] and j < data.shape[1] and k < data.shape[2]:
        data[i][j][k] -= dark[i][j][k]
        flat[i][j][k] -= dark[i][j][k]

        if flat[i][j][k] > 0:
            data[i][j][k] /= flat[i][j][k]
        else:
            data[i][j][k] /= MINIMUM_PIXEL_VALUE

        if data[i][j][k] < clip_min:
            data[i][j][k] = clip_min
        elif data[i][j][k] > clip_max:
            data[i][j][k] = clip_max


@vectorize(["{0}({0},{0})".format(DTYPE)], nopython=True, target="parallel")
def parallel_add_arrays(elem1, elem2):
    return elem1 + elem2


@vectorize("{0}({0},{0},{0})".format(DTYPE), nopython=True, target="parallel")
def parallel_background_correction(data, dark, flat):
    data -= dark
    flat -= dark
    if flat > 0:
        data /= flat
    else:
        data /= MINIMUM_PIXEL_VALUE

    if data < MINIMUM_PIXEL_VALUE:
        return MINIMUM_PIXEL_VALUE
    if data > MAXIMUM_PIXEL_VALUE:
        return MAXIMUM_PIXEL_VALUE
    return data


def get_synchronized_time():
    cuda.synchronize()
    return time.time()


def time_function(func):
    start = get_synchronized_time()
    func()
    return get_synchronized_time() - start


class NumbaImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.warm_up()
        self.lib_name = LIB_NAME
        self.stream = cuda.stream()

    def warm_up(self):
        """
        Give the CUDA functions a chance to compile.
        """
        warm_up_arrays = create_arrays((1, 1, 1), self.dtype)
        add_arrays(*warm_up_arrays[:2])
        background_correction(*warm_up_arrays)

    def clear_cuda_memory(self, split_arrays=[]):

        self.stream.synchronize()

        if not NO_PRINT:
            print("Free bytes before clearing memory:", get_free_bytes())

        if split_arrays:
            for array in split_arrays:
                del array
                array = None
        cuda.current_context().deallocations.clear()
        self.stream.synchronize()

        if NO_PRINT:
            return
        print("Free bytes after clearing memory:", get_free_bytes())

    def _send_arrays_to_gpu(self, cpu_arrays, result_array):

        gpu_arrays = []
        arrays_to_transfer = cpu_arrays + [result_array]

        with cuda.pinned(*arrays_to_transfer):
            for arr in arrays_to_transfer:
                gpu_arrays.append(cuda.to_device(arr, self.stream))
        return gpu_arrays[:-1], gpu_arrays[-1]

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
            start = get_synchronized_time()
            gpu_arrays, gpu_result_array = self._send_arrays_to_gpu(
                self.cpu_arrays[:n_arrs_needed], cpu_result_array
            )
            transfer_time += get_synchronized_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += time_function(
                    lambda: alg(*gpu_arrays[:n_arrs_needed])
                )

            # Time the transfer from GPU to CPU
            transfer_time += time_function(
                lambda: gpu_result_array.copy_to_host(cpu_result_array, self.stream)
            )

            # Free the GPU arrays
            self.clear_cuda_memory(gpu_arrays + [gpu_result_array])

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

                cpu_result_array = np.empty_like(split_cpu_arrays[i])

                try:

                    # Time transferring the segments to the GPU
                    start = get_synchronized_time()
                    gpu_arrays, gpu_result_array = self._send_arrays_to_gpu(
                        split_cpu_arrays, cpu_result_array
                    )
                    transfer_time += get_synchronized_time() - start

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
                    operation_time += time_function(
                        lambda: alg(*gpu_arrays[:n_arrs_needed])
                    )

                transfer_time += time_function(
                    lambda: gpu_result_array.copy_to_host(cpu_result_array, self.stream)
                )

                # Free GPU arrays and partition arrays
                self.clear_cuda_memory(
                    split_cpu_arrays + [gpu_arrays, gpu_result_array]
                )

        if transfer_time > 0 and operation_time > 0:
            self.print_operation_times(operation_time, alg_name, runs, transfer_time)

        return transfer_time + operation_time / runs


for mode in MODES:

    if mode == PARALLEL_VECTORISE_MODE:
        add_arrays = parallel_add_arrays
        background_correction = parallel_background_correction
    elif mode == CUDA_VECTORISE_MODE:
        add_arrays = cuda_vectorise_add_arrays
        background_correction = lambda data, dark, flat: cuda_vectorise_background_correction(
            data, dark, flat, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
        )
    elif mode == CUDA_JIT_MODE:
        add_arrays = cuda_jit_add_arrays
        background_correction = lambda data, dark, flat: cuda_jit_background_correction(
            data, dark, flat, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
        )

    add_arrays_results = []
    background_correction_results = []

    for size in ARRAY_SIZES[:SIZES_SUBSET]:

        imaging_obj = NumbaImplementation(size, DTYPE)

        try:
            avg_add = imaging_obj.timed_imaging_operation(
                N_RUNS, add_arrays, "adding", 2
            )
            avg_bc = imaging_obj.timed_imaging_operation(
                N_RUNS, background_correction, "background correction", 3
            )

            add_arrays_results.append(avg_add)
            background_correction_results.append(avg_bc)

        except cuda.cudadrv.driver.CudaAPIError:
            print("Can't operate on arrays with size:", size)
            print("Free bytes during CUDA error:", get_free_bytes())
            imaging_obj.clear_cuda_memory()
            break

    write_results_to_file([LIB_NAME, mode], ADD_ARRAYS, add_arrays_results)
    write_results_to_file(
        [LIB_NAME, mode], BACKGROUND_CORRECTION, background_correction_results
    )
