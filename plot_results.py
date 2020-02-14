from write_and_read_results import (
    read_results_from_files,
    BACKGROUND_CORRECTION,
    TOTAL_PIXELS,
    ADD_ARRAYS,
)
from matplotlib import pyplot as plt
import pandas as pd

results = read_results_from_files()

print(results)

# Plot Adding Arrays
plt.subplot(2, 2, 1)
plt.title("Average Time Taken To Add Two Arrays")

for key in results.keys():
    print(results[key])
    plt.plot(results[key][ADD_ARRAYS], label=key, marker=".")

plt.yscale("log")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)

## Plot Background Correction Times
plt.subplot(2, 2, 3)
plt.title("Average Time Taken To Do Background Correction")

for key in results.keys():
    plt.plot(results[key][BACKGROUND_CORRECTION], label=key, marker=".")

plt.ylabel("Time Taken")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.yscale("log")
plt.xlabel("Number of Pixels/Elements")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

# Plot Adding Speed Difference
plt.subplot(2, 2, 2)
plt.title("Speed Change For Adding Arrays When Compared With numpy")


def truediv(a, b):
    return a / b


for key in results.keys():
    if key == "numpy":
        continue
    diff = list(map(truediv, results["numpy"][ADD_ARRAYS], results[key][ADD_ARRAYS]))
    plt.plot(diff, label=key, marker=".")
    plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
    plt.xlabel("Number of Pixels/Elements")

# Plot Adding Speed Difference
plt.subplot(2, 2, 4)
plt.title("Speed Change For Background Correction When Compared With numpy")


for key in results.keys():
    if key == "numpy":
        continue
    diff = list(
        map(
            truediv,
            results["numpy"][BACKGROUND_CORRECTION],
            results[key][BACKGROUND_CORRECTION],
        )
    )
    plt.plot(diff, label=key, marker=".")
    plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
    plt.xlabel("Number of Pixels/Elements")

plt.show()
