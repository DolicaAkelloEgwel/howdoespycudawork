from numba import vectorize, cuda
import time

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
)
from imagingtester import num_partitions_needed as number_of_partitions_needed
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

MODES = [PARALLEL_VECTORISE_MODE, CUDA_VECTORISE_MODE, CUDA_JIT_MODE]

if not TEST_PARALLEL_NUMBA:
    MODES = MODES[1:]


def get_free_bytes():
    meminfo = cuda.current_context().get_memory_info()
    return meminfo[0]


FREE_BYTES = get_free_bytes()

num_partitions_needed = lambda cpu_arrays: number_of_partitions_needed(
    cpu_arrays, FREE_BYTES
)


@vectorize(["{0}({0},{0})".format(DTYPE)], target="cuda")
def cuda_vectorise_add_arrays(elem1, elem2):
    return elem1 + elem2


@vectorize("{0}({0},{0},{0})".format(DTYPE), target="cuda")
def cuda_vectorise_background_correction(data, dark, flat):
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
def cuda_jit_background_correction(data, dark, flat):
    i, j, k = cuda.grid(3)

    if i < data.shape[0] and j < data.shape[1] and k < data.shape[2]:
        data[i][j][k] -= dark[i][j][k]
        flat[i][j][k] -= dark[i][j][k]

        if flat[i][j][k] > 0:
            data[i][j][k] /= flat[i][j][k]
        else:
            data[i][j][k] /= MINIMUM_PIXEL_VALUE

        if data[i][j][k] < MINIMUM_PIXEL_VALUE:
            data[i][j][k] = MINIMUM_PIXEL_VALUE
        elif data[i][j][k] > MAXIMUM_PIXEL_VALUE:
            data[i][j][k] = MAXIMUM_PIXEL_VALUE


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


def time_function(func):
    cuda.synchronize()
    start = time.time()
    func()
    cuda.synchronize()
    return time.time() - start


class NumbaImplementation(ImagingTester):
    def __init__(self, size, mode, dtype):
        super().__init__(size, dtype)

        if mode == PARALLEL_VECTORISE_MODE:
            self.add_arrays = parallel_add_arrays
            self.background_correction = parallel_background_correction
        elif mode == CUDA_VECTORISE_MODE:
            self.add_arrays = cuda_vectorise_add_arrays
            self.background_correction = cuda_vectorise_background_correction
        elif mode == CUDA_JIT_MODE:
            self.add_arrays = cuda_jit_add_arrays
            self.background_correction = cuda_jit_background_correction

        self.warm_up()
        self.lib_name = LIB_NAME

    def warm_up(self):
        """
        Give the CUDA functions a chance to compile.
        """
        warm_up_arrays = create_arrays((1, 1, 1), self.dtype)
        self.add_arrays(*warm_up_arrays[:2])
        self.background_correction(*warm_up_arrays)

    def timed_add_arrays(self, runs):

        total_time = 0
        n_partitions_needed = num_partitions_needed(self.cpu_arrays[:2])

        if n_partitions_needed == 1:
            for _ in range(runs):
                total_time += time_function(
                    lambda: self.add_arrays(*self.cpu_arrays[:2])
                )

        else:

            split_arrays = partition_arrays(self.cpu_arrays[:2], n_partitions_needed)

            for i in range(n_partitions_needed):
                cpu_array_segments = [split_array[i] for split_array in split_arrays]
                for _ in range(runs):
                    total_time += time_function(
                        lambda: self.add_arrays(*cpu_array_segments)
                    )

            self.clear_cuda_memory(split_arrays)
        self.clear_cuda_memory()
        self.print_operation_times(total_time, ADD_ARRAYS, runs)
        return total_time / runs

    def timed_background_correction(self, runs):
        total_time = 0

        n_partitions_needed = num_partitions_needed(self.cpu_arrays)

        if n_partitions_needed == 1:

            # gpu_arrays = self._send_arrays_to_gpu(self.cpu_arrays)

            for _ in range(runs):
                total_time += time_function(
                    lambda: self.background_correction(*self.cpu_arrays)
                )

        else:

            split_arrays = partition_arrays(self.cpu_arrays, n_partitions_needed)

            for i in range(n_partitions_needed):

                cpu_array_segments = [split_array[i] for split_array in split_arrays]
                # gpu_arrays = self._send_arrays_to_gpu(self.cpu_arrays)

                for _ in range(runs):
                    total_time += time_function(
                        lambda: self.background_correction(*cpu_array_segments)
                    )

            self.clear_cuda_memory(split_arrays)

        self.clear_cuda_memory()

        self.print_operation_times(total_time, BACKGROUND_CORRECTION, runs)
        return total_time / runs

    def clear_cuda_memory(self, split_arrays=None):

        print("Free bytes before clearing memory", get_free_bytes())

        if split_arrays is not None:
            for array in split_arrays:
                del array
                array = None
        cuda.current_context().deallocations.clear()
        print("Free bytes after clearing memory", get_free_bytes())


for mode in MODES:

    add_arrays = []
    background_correction = []

    for size in ARRAY_SIZES[:SIZES_SUBSET]:

        try:
            imaging_obj = NumbaImplementation(size, mode, DTYPE)

            cuda.synchronize()
            avg_add = imaging_obj.timed_add_arrays(N_RUNS)
            cuda.synchronize()
            avg_bc = imaging_obj.timed_background_correction(N_RUNS)

            add_arrays.append(avg_add)
            background_correction.append(avg_bc)

        except cuda.cudadrv.driver.CudaAPIError:
            print("Can't operate on arrays with size", size)
            break

    write_results_to_file([LIB_NAME, mode], ADD_ARRAYS, add_arrays)
    write_results_to_file(
        [LIB_NAME, mode], BACKGROUND_CORRECTION, background_correction
    )
