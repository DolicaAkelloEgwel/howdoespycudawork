import os
import sys
import time
from math import ceil

import yaml

import numpy as np

MINIMUM_PIXEL_VALUE = 1e-9
MAXIMUM_PIXEL_VALUE = 200  # this isn't true but it makes random easier

PRINT_INFO = None
N_RUNS = None
SIZES_SUBSET = None
DTYPE = None
TEST_PARALLEL_NUMBA = None
USE_NONPINNED_MEMORY = None

# Retrieve the benchmark parameters from the yaml file
with open(os.path.join(os.getcwd(), "benchmarkparams.yaml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    PRINT_INFO = params["print_info"]
    N_RUNS = params["runs"]
    DTYPE = params["dtype"]
    SIZES_SUBSET = params["sizes_subset"]
    TEST_PARALLEL_NUMBA = params["test_parallel_numba"]
    USE_NONPINNED_MEMORY = params["use_nonpinned_memory"]


def create_arrays(size_tuple, dtype):
    """
    Create three arrays of a given size containing random values.
    :param size_tuple: The desired size of the arrays.
    :param dtype: The desired data type of the arrays.
    :return: Three arrays containing values between the "minimum" and "maximum" pixel values.
    """
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
        self.dtype = dtype
        self.create_arrays(size, dtype)

    def create_arrays(self, size_tuple, dtype):
        start = time.time()
        self.cpu_arrays = create_arrays(size_tuple, dtype)
        end = time.time()
        print_array_creation_time(end - start)

    def warm_up(self):
        pass

    def print_operation_times(
        self, operation_time, operation_name, runs, transfer_time=None
    ):
        """
        Print the time spent doing performing a calculation and the time spent transferring arrays.
        :param operation_name: The name of the imaging algorithm.
        :param operation_time: The time the GPU took doing the calculations.
        :param runs: The number of runs used to obtain the average operation time.
        :param transfer_time: The time spent transferring the arrays to and from the GPU.
        """
        if not PRINT_INFO:
            return
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


def print_array_creation_time(time):
    """
    Print the array creation time. Generating large random arrays can take a while.
    :param time: Time taken to create the array.
    """
    if not PRINT_INFO:
        return
    print("Array creation time: %ss" % time)


def memory_needed_for_arrays(cpu_arrays):
    return sum([sys.getsizeof(arr) for arr in cpu_arrays])


def partition_arrays(cpu_arrays, n_partitions):
    return [np.array_split(cpu_array, n_partitions) for cpu_array in cpu_arrays]


def num_partitions_needed(cpu_arrays, free_bytes):
    return int(ceil(memory_needed_for_arrays(cpu_arrays) * 1.0 / free_bytes))
