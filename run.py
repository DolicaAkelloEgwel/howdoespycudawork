import argparse
from subprocess import call

TEST = "test_"
libraries = ["numpy", "numba", "cupy", "pycuda"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--runs",
    type=int,
    default=20,
    help="The number of runs that should be carried out for the imaging procedures in order to obtain an average performance time.",
)
parser.add_argument(
    "--sizes_subset",
    type=int,
    default=5,
    choices=range(1, 5),
    help="How many of the first X elements in the list of sizes will be used for testing performance.",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="float32",
    choices=["float16", "float32", "float64"],
    help="The float datatype that will be used for the CPU/GPU arrays. Higher precision will require more GPU transfer.",
)
parser.add_argument(
    "--printstuff",
    default=False,
    action="store_true",
    help="Whether or not you want to see print statements every time something happens.",
)
args = parser.parse_args()

runs = str(args.runs)
sizes_subset = str(args.sizes_subset)
dtype = args.dtype
print_stuff = str(args.printstuff)

for lib in libraries:
    call(["python", TEST + lib + ".py", runs, sizes_subset, dtype, print_stuff])
