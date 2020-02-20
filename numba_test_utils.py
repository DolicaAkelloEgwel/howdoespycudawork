from numba import cuda, vectorize

from imagingtester import ImagingTester, PRINT_INFO, DTYPE, MINIMUM_PIXEL_VALUE

LIB_NAME = "numba"


def get_free_bytes():
    return cuda.current_context().get_memory_info()[0]


def get_used_bytes():
    return cuda.current_context().get_memory_info()[1] - get_free_bytes()


STREAM = cuda.stream()


def create_vectorise_add_arrays(target):
    @vectorize(["{0}({0},{0})".format(DTYPE)], target=target)
    def add_arrays(elem1, elem2):
        return elem1 + elem2

    return add_arrays


def create_vectorise_background_correction(target):
    @vectorize("{0}({0},{0},{0},{0},{0})".format(DTYPE), target=target)
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

    return background_correction


class NumbaImplementation(ImagingTester):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)
        self.warm_up()
        self.lib_name = LIB_NAME

    def warm_up(self):
        pass

    def get_time(self):
        pass

    def time_function(self, func):
        start = self.get_time()
        func()
        return self.get_time() - start

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

    def _send_arrays_to_gpu(self, cpu_arrays):

        gpu_arrays = []
        arrays_to_transfer = cpu_arrays

        with cuda.pinned(*arrays_to_transfer):
            for arr in arrays_to_transfer:
                gpu_arrays.append(cuda.to_device(arr, STREAM))

        return gpu_arrays

    def timed_imaging_operation(self, runs, alg, alg_name, n_arrs_needed):
        pass
