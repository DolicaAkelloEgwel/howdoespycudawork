from write_and_read_results import (
    read_results_from_files,
    BACKGROUND_CORRECTION,
    TOTAL_PIXELS,
    ADD_ARRAYS,
)
from matplotlib import pyplot as plt

results = read_results_from_files()

nonmedian_colours = dict()
median_colours = dict()


# Plot Adding Arrays
plt.subplot(3, 2, 2)
plt.title("Average Time Taken To Add Two Arrays")

for key in results.keys():
    try:
        p = plt.plot(results[key][ADD_ARRAYS], label=key, marker=".")
    except KeyError:
        continue
    nonmedian_colours[key] = p[-1].get_color()

plt.yscale("log")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)

## Plot Background Correction Times
plt.subplot(3, 2, 4)
plt.title("Average Time Taken To Do Background Correction")

for key in results.keys():
    try:
        plt.plot(
            results[key][BACKGROUND_CORRECTION],
            label=key,
            marker=".",
            color=nonmedian_colours[key],
        )
    except KeyError:
        continue

plt.ylabel("Time Taken")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.yscale("log")
plt.xlabel("Number of Pixels/Elements")
plt.legend()

## Plot Median Filter
plt.subplot(3, 2, 6)
plt.title("Average Time Taken To Do Median Filter")

for key in [
    "scipy",
    "pycuda sourcemodule",
    "cupy with pinned memory",
    "cupy without pinned memory",
]:
    try:
        p = plt.plot(results[key]["median filter"], label=key, marker=".")
        median_colours[key] = p[-1].get_color()
    except KeyError:
        continue

plt.yscale("log")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.legend()


def truediv(a, b):
    return a / b


# Plot Adding Speed Difference
plt.subplot(3, 2, 1)
plt.title("Speed Change For Adding Arrays When Compared With numpy")

for key in results.keys():
    if key == "numpy":
        continue
    try:
        diff = list(
            map(truediv, results["numpy"][ADD_ARRAYS], results[key][ADD_ARRAYS])
        )
        plt.plot(diff, label=key, marker=".", color=nonmedian_colours[key])
    except KeyError:
        continue
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.xlabel("Number of Pixels/Elements")

# Plot Background Correction Speed Difference
plt.subplot(3, 2, 3)
plt.title("Speed Change For Background Correction When Compared With numpy")

for key in results.keys():
    if key == "numpy":
        continue
    try:
        diff = list(
            map(
                truediv,
                results["numpy"][BACKGROUND_CORRECTION],
                results[key][BACKGROUND_CORRECTION],
            )
        )
    except KeyError:
        continue
    print(key, diff)
    plt.plot(diff, label=key, marker=".", color=nonmedian_colours[key])
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.xlabel("Number of Pixels/Elements")

# Plot Adding Speed Difference
plt.subplot(3, 2, 5)
plt.title("Speed Change For Median Filter When Compared With scipy")

for key in results.keys():
    if key == "scipy":
        continue
    try:
        diff = list(
            map(
                truediv,
                results["scipy"]["median filter"],
                results[key]["median filter"],
            )
        )
        plt.plot(diff, label=key, marker=".", color=median_colours[key])
    except KeyError:
        continue
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.xlabel("Number of Pixels/Elements")

plt.show()
