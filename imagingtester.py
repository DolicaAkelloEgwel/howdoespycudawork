import argparse

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

RESULTS_DIR = "results/"
SPACE_STRING = " "
ADD_ARRAYS = "add arrays"
BACKGROUND_CORRECTION = "background correction"


parser = argparse.ArgumentParser()
parser.add_argument(
    "-runs",
    type=int,
    default=20,
    help="The number of runs that should be carried out for the imaging procedures in order to obtain an average performance time.",
)
parser.add_argument(
    "-sizes_subset",
    type=int,
    default=len(ARRAY_SIZES),
    choices=range(1, len(ARRAY_SIZES)),
    help="How many of the first X elements in the list of sizes {} will be used for testing performance.".format(
        ARRAY_SIZES
    ),
)
parser.add_argument(
    "-dtype",
    type=str,
    default="float32",
    choices=["float16", "float32", "float64"],
    help="The float datatype that will be used for the CPU/GPU arrays. Higher precision will require more GPU transfer.",
)
parser.add_argument(
    "-printstuff",
    type=bool,
    default=True,
    help="Whether or not you want to see print statements every time something happens.",
)
args = parser.parse_args()

N_RUNS = args.runs
SIZES_SUBSET = args.sizes_subset
DTYPE = args.dtype
NO_PRINT = not args.printstuff


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
        self.dtype = dtype
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
        if NO_PRINT:
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
    if NO_PRINT:
        return
    print("Array creation time: %ss" % time)


def write_results_to_file(name_list, results):
    """
    Write the timing results to a file. in the "results" directory.
    :param name_list:
    :param results:
    """
    name = SPACE_STRING.join(name_list)
    filename = name.replace(" ", "_")
    with open(RESULTS_DIR + filename, "w") as f:
        f.write(name)
        f.write("\n")
        for val in results:
            f.write(str(val) + "\n")
