import time

import numpy as np

MINIMUM_PIXEL_VALUE = 1e-9
MAXIMUM_PIXEL_VALUE = 1e9


class ImagingTester:
    def __init__(self, size):
        self.create_arrays(size)

    def create_arrays(self, size_tuple):
        self.cpu_arrays = [
            np.random.uniform(
                low=MINIMUM_PIXEL_VALUE, high=MAXIMUM_PIXEL_VALUE, size=size_tuple
            ).astype("float32")
            for _ in range(3)
        ]

    def timed_add_arrays(self):
        pass

    def timed_background_correction(self):
        pass


class NumpyImplementation(ImagingTester):
    def __init__(self, size):
        super().__init__(size)

    def timed_add_arrays(self, reps):
        arr1, arr2 = self.cpu_arrays[:2]
        total_time = 0
        for _ in range(reps):
            start = time.time()
            ###
            np.add(arr1, arr2)
            ###
            total_time += time.time() - start
        return total_time / reps

    def timed_background_correction(self, reps):
        data, dark, flat = self.cpu_arrays
        total_time = 0
        for _ in range(reps):
            start = time.time()
            ###
            np.subtract(data, dark, out=data)
            np.subtract(flat, dark, out=flat)
            np.true_divide(data, flat, out=data)
            ###
            total_time += time.time() - start
        return total_time / reps
