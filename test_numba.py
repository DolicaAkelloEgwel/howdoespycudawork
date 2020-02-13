from numba import vectorize
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
)


@vectorize(["float(float, float)"], target="cuda")
def cuda_add_arrays(elem1, elem2):
    return elem1 + elem2


@vectorize("float(float,float,float)", target="cuda")
def cuda_background_correction(data, dark, flat):
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


@vectorize(["float(float, float)"], nopython=True, target="parallel")
def parallel_add_arrays(elem1, elem2):
    return elem1 + elem2


@vectorize("float(float,float,float)", nopython=True, target="parallel")
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


class CudaNumbaImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.warm_up()

    def warm_up(self):
        """
        Give the CUDA functions a chance to compile.
        """
        warm_up_arrays = create_arrays((1, 1, 1), self.dtype)
        cuda_add_arrays(*warm_up_arrays[:2])
        cuda_background_correction(*warm_up_arrays)

    @staticmethod
    def time_function(func):
        start = time.time()
        func()
        return time.time() - start

    def timed_add_arrays(self, runs):
        total_time = 0

        for _ in range(runs):
            total_time += self.time_function(
                lambda: cuda_add_arrays(*self.cpu_arrays[:2])
            )
        return total_time / runs

    def timed_background_correction(self, runs):
        total_time = 0

        for _ in range(runs):
            total_time += self.time_function(
                lambda: cuda_background_correction(*self.cpu_arrays)
            )
        return total_time


class ParallelNumbaImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.warm_up()

    def warm_up(self):
        """
        Give the CUDA functions a chance to compile.
        """
        warm_up_arrays = create_arrays((1, 1, 1), self.dtype)
        parallel_add_arrays(*warm_up_arrays[:2])
        parallel_background_correction(*warm_up_arrays)

    def time_function(self, func):
        start = time.time()
        func()
        return time.time() - start

    def timed_add_arrays(self, runs):
        total_time = 0

        for _ in range(runs):
            total_time += self.time_function(
                lambda: parallel_add_arrays(*self.cpu_arrays[:2])
            )
        return total_time / runs

    def timed_background_correction(self, runs):
        total_time = 0

        for _ in range(runs):
            total_time += self.time_function(
                lambda: parallel_background_correction(*self.cpu_arrays)
            )
        return total_time


implementations = [ParallelNumbaImplementation, CudaNumbaImplementation]
implementation_names = {
    ParallelNumbaImplementation: "parallel numba",
    CudaNumbaImplementation: "cuda numba",
}
results_values = {ParallelNumbaImplementation: dict(), CudaNumbaImplementation: dict()}

for impl in [ParallelNumbaImplementation, CudaNumbaImplementation]:

    add_arrays = []
    background_correction = []

    for size in ARRAY_SIZES:

        try:
            imaging_obj = impl(size, DTYPE)

            avg_add = imaging_obj.timed_add_arrays(N_RUNS)
            avg_bc = imaging_obj.timed_background_correction(N_RUNS)

            add_arrays.append(avg_add)
            background_correction.append(avg_bc)

        except numba.cuda.cudadrv.driver.CudaAPIError as e:
            print("Unable to carry out calculation on array of size", size)
            print(e)
            break

    write_results_to_file([implementation_names[impl], ADD_ARRAYS], add_arrays)
    write_results_to_file(
        [implementation_names[impl], BACKGROUND_CORRECTION], background_correction
    )
