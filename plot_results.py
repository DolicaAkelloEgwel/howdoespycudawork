from write_and_read_results import (
    read_results_from_files,
    BACKGROUND_CORRECTION,
    TOTAL_PIXELS,
    ADD_ARRAYS,
)
from matplotlib import pyplot as plt

results = read_results_from_files()

# Plot Adding Arrays
plt.subplot(1, 2, 1)
plt.title("Average Time Taken To Add Two Arrays")

for key in results.keys():
    if ADD_ARRAYS in key:
        plt.plot(results[key], label=key, marker=".")

plt.yscale("log")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.legend()

## Plot Background Correction Times
plt.subplot(1, 2, 2)
plt.title("Average Time Taken To Do Background Correction")

for key in results.keys():
    if BACKGROUND_CORRECTION in key:
        plt.plot(results[key], label=key, marker=".")

plt.ylabel("Time Taken")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.yscale("log")
plt.xlabel("Number of Pixels/Elements")
plt.legend()

plt.show()
