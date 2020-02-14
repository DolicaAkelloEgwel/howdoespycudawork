from write_and_read_results import read_results_from_files, BACKGROUND_CORRECTION
from matplotlib import pyplot as plt

results = read_results_from_files()

for key in results.keys():
    if BACKGROUND_CORRECTION in key:
        plt.plot(results[key], label=key)

plt.yscale("log")
plt.legend()
plt.show()
