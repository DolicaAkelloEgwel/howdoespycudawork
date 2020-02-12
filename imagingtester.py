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

    def timed_add_arrays(self, runs):
        pass

    def timed_background_correction(self, runs):
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
            norm_divide = np.subtract(flat, dark)
            norm_divide[norm_divide == 0] = MINIMUM_PIXEL_VALUE
            np.subtract(data, dark, out=data)
            np.true_divide(data, norm_divide, out=data)
            np.clip(data, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE, out=data)
            ###
            total_time += time.time() - start
        return total_time / reps


ARRAY_SIZES = [
    (10, 100, 500),
    (100, 100, 500),
    (100, 1000, 500),
    (1000, 1000, 500),
    (1500, 1500, 500),
]
TOTAL_PIXELS = [x * y * z for x, y, z in ARRAY_SIZES]
