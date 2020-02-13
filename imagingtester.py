import numpy as np

MINIMUM_PIXEL_VALUE = 1e-9
MAXIMUM_PIXEL_VALUE = 1e9

ARRAY_SIZES = [
    (10, 100, 1000),
    (100, 100, 1000),
    (100, 1000, 1000),
    (1000, 1000, 1000),
    (1500, 1500, 1000),
]
TOTAL_PIXELS = [x * y * z for x, y, z in ARRAY_SIZES]

N_RUNS = 20


def create_arrays(size_tuple, dtype):
    return [
        np.random.uniform(
            low=MINIMUM_PIXEL_VALUE, high=MAXIMUM_PIXEL_VALUE, size=size_tuple
        ).astype(dtype)
        for _ in range(3)
    ]


class ImagingTester:
    def __init__(self, size, dtype):
        self.cpu_arrays = None
        self.lib_name = None
        self.create_arrays(size, dtype)

    def create_arrays(self, size_tuple, dtype):
        self.cpu_arrays = create_arrays(size_tuple, dtype)

    def warm_up(self):
        pass

    def timed_add_arrays(self, runs):
        pass

    def timed_background_correction(self, runs):
        pass

    def print_operation_times(
        self, operation_time, operation_name, runs, transfer_time
    ):
        """
        Print the time spent doing performing a calculation and the time spent transferring arrays.
        :param operation_name: The name of the imaging algorithm.
        :param operation_time: The time the GPU took doing the calculations.
        :param runs: The number of runs used to obtain the average operation time.
        :param transfer_time: The time spent transferring the arrays to and from the GPU.
        """
        if transfer_time is not None:
            print(
                "With %s transferring arrays of size %s took %ss and %s took an average of %ss over %s runs."
                % (
                    self.lib_name,
                    self.cpu_arrays[0].shape,
                    transfer_time,
                    operation_name,
                    operation_time / runs,
                    runs,
                )
            )
        else:
            print(
                "With %s carrying out %s on arrays of size %s took an average of %ss over %s runs."
                % (
                    self.lib_name,
                    operation_name,
                    self.cpu_arrays[0].shape,
                    operation_time / runs,
                    runs,
                )
            )


def write_results_to_file(filename):
    with open(filename, "w") as f:
        pass
