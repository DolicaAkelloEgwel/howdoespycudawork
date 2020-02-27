## GPU Benchmarking
A script for comparing `cupy`, `numba`, and `pycuda` when performing Background Correction.

### Usage
Execute the `run` file (requires bash). This will perform all the tests and generate a plot showing execution times and performance change when compared to `numpy`.

#### Parameters
The `benchmarkparams.yml` file allows you to change the benchmark configurations. The options that can be changed are:  
- `runs` (int) - Number of runs performed in order to determine the average execution time.  
- `sizes_subset` (int) - The number of different array sizes to use for benchmarking. It's currently configured to test five different array sizes ordered from smallest to largest. Setting this to 0 <= X <= 5 will cause it to only use the first X sizes.  
- `dtype` (string) - The data type used for the array. Can be either float32 or float 64.  
- `print_info` (bool) - Whether or not you want to see some messages pop up when an algorithm finishes execution and how long it took to transfer + process the array.  
- `test_parallel_numba` (bool) - Whether or not include parallel `numba` in the benchmarking.  
- `free memory factor` (float) - A value in the range (0,1] that will be used to determine how to "correct" the free memory reported by the different libraries so that partitioning larger arrays doesn't fail.  
- `use_cupy_nonpinned_memory` (bool) - Whether or not to benchmark `cupy` without pinned memory. The absence of pinned memory tends to make transfer speeds much worse.  
![plotz](https://github.com/DolicaAkelloEgwel/howdoespycudawork/blob/master/Figure_1.png "")  