from numba import vectorize, cuda
import numba
import time

from imagingtester import (
    ImagingTester,
    ARRAY_SIZES,
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
    write_results_to_file,
    N_RUNS,
    DTYPE,
    create_arrays,
    SIZES_SUBSET,
)

LIB_NAME = "numba"

PARALLEL_VECTORISE_MODE = "parallel vectorise"
CUDA_VECTORISE_MODE = "cuda vectorise"
CUDA_JIT_MODE = "cuda jit"
MODES = [PARALLEL_VECTORISE_MODE, CUDA_VECTORISE_MODE, CUDA_JIT_MODE]


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
        cuda_vectorise_add_arrays(*warm_up_arrays[:2])
        cuda_vectorise_background_correction(*warm_up_arrays)

    @staticmethod
    def time_function(func):
        start = time.time()
        func()
        return time.time() - start

    def timed_add_arrays(self, runs):
        total_time = 0

        for _ in range(runs):
            total_time += self.time_function(
                lambda: self.add_arrays(*self.cpu_arrays[:2])
            )
        self.print_operation_times(total_time, ADD_ARRAYS, runs)
        return total_time / runs

    def timed_background_correction(self, runs):
        total_time = 0

        for _ in range(runs):
            total_time += self.time_function(
                lambda: self.background_correction(*self.cpu_arrays)
            )
        self.print_operation_times(total_time, BACKGROUND_CORRECTION, runs)
        return total_time


for mode in MODES:

    add_arrays = []
    background_correction = []

    for size in ARRAY_SIZES[:SIZES_SUBSET]:

        try:
            imaging_obj = NumbaImplementation(size, mode, DTYPE)

            avg_add = imaging_obj.timed_add_arrays(N_RUNS)
            avg_bc = imaging_obj.timed_background_correction(N_RUNS)

            add_arrays.append(avg_add)
            background_correction.append(avg_bc)

        except numba.cuda.cudadrv.driver.CudaAPIError as e:
            print("Unable to carry out calculation on array of size", size)
            print(e)
            break

    write_results_to_file([LIB_NAME, mode, ADD_ARRAYS], add_arrays)
    write_results_to_file(
        [LIB_NAME, mode, BACKGROUND_CORRECTION], background_correction
    )
