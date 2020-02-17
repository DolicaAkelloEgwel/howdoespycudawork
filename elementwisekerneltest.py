import time

import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit

from pycuda.elementwise import ElementwiseKernel

from imagingtester import (
    create_arrays,
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
    DTYPE,
    N_RUNS,
)
from write_and_read_results import (
    ARRAY_SIZES,
    write_results_to_file,
    BACKGROUND_CORRECTION,
)

LIB_NAME = "pycuda"

# Initialise PyCuda Driver
drv.init()
drv.Device(0).make_context()

# Create an element-wise Background Correction Function
BackgroundCorrectionKernel = ElementwiseKernel(
    arguments="float * data, float * flat, const float * dark, const float MINIMUM_PIXEL_VALUE, const float MAXIMUM_PIXEL_VALUE",
    operation="flat[i] -= dark[i];"
    "if (flat[i] <= 0) flat[i] = MINIMUM_PIXEL_VALUE;"
    "data[i] -= dark[i];"
    "data[i] /= flat[i];"
    "if (flat[i] > MAXIMUM_PIXEL_VALUE) flat[i] = MAXIMUM_PIXEL_VALUE;"
    "if (flat[i] < MINIMUM_PIXEL_VALUE) flat[i] = MINIMUM_PIXEL_VALUE;",
    name="BackgroundCorrectionKernel",
)

elementwise_background_correction = lambda data, flat, dark: BackgroundCorrectionKernel(
    data, flat, dark, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
)

# Create an element-wise Add Array Function
AddArraysKernel = ElementwiseKernel(
    arguments="float * arr1, float * arr2",
    operation="arr1[i] += arr2[i];",
    name="AddArraysKernel",
)


def send_arrays_to_gpu(cpu_arrays):
    return [gpuarray.to_gpu(cpu_array) for cpu_array in cpu_arrays]


def free_memory_pool(gpu_arrays):
    for gpu_array in gpu_arrays:
        gpu_array.gpudata.free()


def get_synchronized_time():
    drv.Context.synchronize()
    return time.time()


def time_function(func):
    start = get_synchronized_time()
    func()
    end = get_synchronized_time()
    return end - start


def find_average_time(func, runs):
    total_time = 0
    for _ in range(runs):
        total_time += time_function(func)
    return total_time / runs


# Warm-up Kernels
warm_up_size = (1, 1, 1)
cpu_arrays = create_arrays(warm_up_size, DTYPE)
gpu_arrays = [gpuarray.to_gpu(array) for array in cpu_arrays]
BackgroundCorrectionKernel(
    gpu_arrays[0],
    gpu_arrays[1],
    gpu_arrays[2],
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
)
AddArraysKernel(gpu_arrays[0], gpu_arrays[1])

background_correction_times = []
add_arrays = []


def time_background_correction_and_transfer(gpu_arrays):

    # Send the arrays to GPU and time it
    start = get_synchronized_time()
    gpu_arrays = [gpuarray.to_gpu(cpu_array) for cpu_array in cpu_arrays]
    end = get_synchronized_time()
    cpu_to_gpu_time = end - start

    # Carry out background correction
    operation_time = find_average_time(
        lambda: elementwise_background_correction(
            gpu_arrays[0], gpu_arrays[1], gpu_arrays[2]
        ),
        N_RUNS,
    )

    # Time retrieving the result
    start = get_synchronized_time()
    arr = gpu_arrays[0].get()
    end = get_synchronized_time()
    gpu_to_cpu_time = end - start

    return cpu_to_gpu_time + operation_time + gpu_to_cpu_time


for size in ARRAY_SIZES:

    cpu_arrays = create_arrays(size, DTYPE)

    try:

        background_correction_times.append(
            time_background_correction_and_transfer(gpu_arrays)
        )
        free_memory_pool(gpu_arrays)

    except drv.MemoryError as e:
        print(e)
        print("Unable to make GPU arrays with size", size)
        break

write_results_to_file(
    [LIB_NAME, "elementwise kernel"], BACKGROUND_CORRECTION, background_correction_times
)
