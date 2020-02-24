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
    ADD_ARRAYS,
)


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


def time_background_correction_and_transfer(cpu_arrays, background_correction):

    cpu_to_gpu_time, gpu_arrays = timed_send_arrays_to_gpu(cpu_arrays)

    # Carry out background correction
    operation_time = find_average_time(
        lambda: background_correction(gpu_arrays[0], gpu_arrays[1], gpu_arrays[2]),
        N_RUNS,
    )

    gpu_to_cpu_time = timed_get_array_from_gpu(gpu_arrays[0])
    free_memory_pool(gpu_arrays)

    return cpu_to_gpu_time + operation_time + gpu_to_cpu_time


def time_adding_arrays_and_transfer(cpu_arrays, add_arrays):

    cpu_to_gpu_time, gpu_arrays = timed_send_arrays_to_gpu(cpu_arrays[:2])
    operation_time = find_average_time(lambda: add_arrays(*gpu_arrays), N_RUNS)
    gpu_to_cpu_time = timed_get_array_from_gpu(gpu_arrays[0])
    free_memory_pool(gpu_arrays)

    return cpu_to_gpu_time + operation_time + gpu_to_cpu_time


background_correction_elementwise_times = []
add_arrays_elementwise_times = []
background_correction_sourcemodule_times = []
add_arrays_sourcemodule_times = []


for size in ARRAY_SIZES:

    cpu_arrays = create_arrays(size, DTYPE)

    try:

        background_correction_elementwise_times.append(
            time_background_correction_and_transfer(
                cpu_arrays, elementwise_background_correction
            )
        )
        add_arrays_elementwise_times.append(
            time_adding_arrays_and_transfer(cpu_arrays, AddArraysKernel)
        )

        # background_correction_sourcemodule_times.append(
        #     time_background_correction_and_transfer(
        #         cpu_arrays, sourcemodule_background_correction
        #     )
        # )
        # add_arrays_sourcemodule_times.append(
        #     time_adding_arrays_and_transfer(cpu_arrays, sourcemodule_add_arrays)
        # )

    except drv.MemoryError as e:
        print(e)
        print("Unable to make GPU arrays with size", size)
        break

write_results_to_file(
    [LIB_NAME, "elementwise kernel"],
    BACKGROUND_CORRECTION,
    background_correction_elementwise_times,
)
write_results_to_file(
    [LIB_NAME, "elementwise kernel"], ADD_ARRAYS, add_arrays_elementwise_times
)
write_results_to_file(
    [LIB_NAME, "sourcemodule"],
    BACKGROUND_CORRECTION,
    background_correction_sourcemodule_times,
)
write_results_to_file(
    [LIB_NAME, "sourcemodule"], ADD_ARRAYS, add_arrays_sourcemodule_times
)


drv.Context.pop()
