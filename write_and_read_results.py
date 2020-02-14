import os

SPACE_STRING = " "
RESULTS_DIR = "results/"


def write_results_to_file(lib_and_mode, alg, results):
    """
    Write the timing results to a file. in the "results" directory.
    :param lib_and_mode:
    :param results:
    """
    name = SPACE_STRING.join(lib_and_mode)
    filename = name.replace(SPACE_STRING, "_")
    with open(RESULTS_DIR + filename, "w+") as f:
        f.write(name)
        f.write("\n")
        f.write(alg)
        f.write("\n")
        for val in results:
            f.write(str(val) + "\n")


def read_results_from_files():

    results = dict()

    for filename in os.listdir(os.path.join(os.getcwd(), RESULTS_DIR)):
        with open(RESULTS_DIR + filename) as f:
            name = f.readline()
            alg = f.readline()
            results[name] = dict()
            results[name][alg] = [float(line) for line in f.readlines()]

    return results


ADD_ARRAYS = "add arrays"
BACKGROUND_CORRECTION = "background correction"
ARRAY_SIZES = [
    (10, 100, 1000),
    (100, 100, 1000),
    (100, 1000, 1000),
    (1000, 1000, 1000),
    (1500, 1500, 1000),
]
TOTAL_PIXELS = [x * y * z for x, y, z in ARRAY_SIZES]
