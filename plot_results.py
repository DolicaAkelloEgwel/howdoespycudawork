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
ax = plt.subplot(1, 2, 1)
plt.title("Average Time Taken To Add Two Arrays")

for key in results.keys():
    print(results[key])
    plt.plot(results[key][ADD_ARRAYS], label=key, marker=".")

plt.yscale("log")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

## Plot Background Correction Times
plt.subplot(1, 2, 2)
plt.title("Average Time Taken To Do Background Correction")

for key in results.keys():
    plt.plot(results[key][BACKGROUND_CORRECTION], label=key, marker=".")

plt.ylabel("Time Taken")
plt.xticks(range(len(TOTAL_PIXELS)), TOTAL_PIXELS)
plt.yscale("log")
plt.xlabel("Number of Pixels/Elements")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

plt.show()
