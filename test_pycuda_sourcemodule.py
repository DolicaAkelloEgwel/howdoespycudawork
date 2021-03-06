from pycuda.compiler import SourceModule
import pycuda.driver as drv
import numpy as np

from imagingtester import (
    DTYPE,
    create_arrays,
    SIZES_SUBSET,
    num_partitions_needed,
    N_RUNS,
    get_array_partition_indices,
    FILTER_SIZE,
)
from pycuda_test_utils import (
    PyCudaImplementation,
    get_free_bytes,
    get_time,
    time_function,
    free_memory_pool,
)
from write_and_read_results import ARRAY_SIZES, write_results_to_file

LIB_NAME = "pycuda"
mode = "sourcemodule"

median_filter_module = SourceModule(
    """
__device__ float find_median(float* neighb_array, const int N)
{
    int i, j;
    float key;

    for (i = 1; i < N; i++)
    {
        key = neighb_array[i];
        j = i - 1;

        while (j >= 0 && neighb_array[j] > key)
        {
            neighb_array[j + 1] = neighb_array[j];
            j = j - 1;
        }
        neighb_array[j + 1] = key;
    }
    return neighb_array[N / 2];
}
__global__ void median_filter(float* data_array, const float* padded_array, const int N_IMAGES, const int X, const int Y, const int filter_height, const int filter_width)
{
    unsigned int id_img = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int id_x = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int id_y = blockIdx.z*blockDim.z + threadIdx.z;
    unsigned int n_counter = 0;
    unsigned int img_size =  X * Y;
    unsigned int padded_img_size =  (X + filter_height - 1) * (Y + filter_width - 1);
    unsigned int padded_img_width =  X + filter_height - 1;

    float neighb_array[25];

    if ((id_img < N_IMAGES) && (id_x < X) && (id_y < Y))
    {
        for (int i = id_x; i < id_x + filter_height; i++)
        {
            for (int j = id_y; j < id_y + filter_width; j++)
            {
                neighb_array[n_counter] = padded_array[(id_img * padded_img_size) + (i * padded_img_width) + j];
                n_counter += 1;
            }
        }

        data_array[(id_img * img_size) + (id_x * X) + id_y] = find_median(neighb_array, filter_height * filter_width);
    }
}
"""
)  #

median_filter = median_filter_module.get_function("median_filter")


def pycuda_median_filter(data, padded_data, filter_height, filter_width):
    median_filter(
        data,
        padded_data,
        np.int32(data.shape[0]),
        np.int32(data.shape[1]),
        np.int32(data.shape[2]),
        np.int32(filter_height),
        np.int32(filter_width),
        block=(10, 10, 10),
    )


class PyCudaSourceModuleImplementation(PyCudaImplementation):
    def __init__(self, size, dtype):

        super().__init__(size, dtype)
        self.warm_up()

    def warm_up(self):
        warm_up_size = (2, 2, 2)
        cpu_data_array = create_arrays(warm_up_size, DTYPE)[0]

        filter_height = 3
        filter_width = 3
        pad_height = filter_height // 2
        pad_width = filter_width // 2

        padded_cpu_array = np.pad(
            cpu_data_array,
            pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width)),
            mode="reflect",
        )
        gpu_data_array, gpu_padded_array = self._send_arrays_to_gpu(
            [cpu_data_array, padded_cpu_array]
        )
        pycuda_median_filter(
            gpu_data_array, gpu_padded_array, filter_height, filter_width
        )

    def timed_median_filter(self, runs, filter_size):

        n_partitions_needed = num_partitions_needed(
            self.cpu_arrays[:1], get_free_bytes()
        )

        transfer_time = 0
        operation_time = 0

        pad_height = filter_size[1] // 2
        pad_width = filter_size[0] // 2

        filter_height = filter_size[0]
        filter_width = filter_size[1]

        if n_partitions_needed == 1:

            # Time transfer from CPU to GPU (and padding creation)
            start = get_time()
            cpu_padded_array = np.pad(
                self.cpu_arrays[0],
                pad_width=((0, 0), (pad_width, pad_width), (pad_height, pad_height)),
            )
            gpu_data_array, gpu_padded_array = self._send_arrays_to_gpu(
                [self.cpu_arrays[0], cpu_padded_array]
            )
            transfer_time += get_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += time_function(
                    lambda: pycuda_median_filter(
                        gpu_data_array, gpu_padded_array, filter_height, filter_width
                    )
                )

            # Time the transfer from GPU to CPU
            transfer_time += time_function(lambda: gpu_data_array[0].get_async)

            # Free the GPU arrays
            free_memory_pool([gpu_data_array, gpu_padded_array])

        else:

            n_partitions_needed = num_partitions_needed(
                self.cpu_arrays[:1], get_free_bytes()
            )

            indices = get_array_partition_indices(
                self.cpu_arrays[0].shape[0], n_partitions_needed
            )

            for i in range(n_partitions_needed):

                # Retrieve the segments used for this iteration of the operation
                split_cpu_array = self.cpu_arrays[0][indices[i][0] : indices[i][1] :, :]

                # Time transferring the segments to the GPU
                start = get_time()
                cpu_padded_array = np.pad(
                    split_cpu_array,
                    pad_width=(
                        (0, 0),
                        (pad_width, pad_width),
                        (pad_height, pad_height),
                    ),
                )
                gpu_data_array, gpu_padded_array = self._send_arrays_to_gpu(
                    [split_cpu_array, cpu_padded_array]
                )
                transfer_time += get_time() - start

                # Carry out the operation on the slices
                for _ in range(runs):
                    operation_time += time_function(
                        lambda: pycuda_median_filter(
                            gpu_data_array,
                            gpu_padded_array,
                            filter_height,
                            filter_width,
                        )
                    )

                transfer_time += time_function(lambda: gpu_data_array.get_async)

                # Free the GPU arrays
                free_memory_pool([gpu_data_array, gpu_padded_array])

        if transfer_time > 0 and operation_time > 0:
            self.print_operation_times(
                operation_time, "median filter", runs, transfer_time
            )

        self.synchronise()

        return transfer_time + operation_time / N_RUNS


median_filter_results = []

for size in ARRAY_SIZES[:SIZES_SUBSET]:

    obj = PyCudaSourceModuleImplementation(size, DTYPE)
    avg_median = obj.timed_median_filter(N_RUNS, FILTER_SIZE)

    if avg_median > 0:
        median_filter_results.append(avg_median)

write_results_to_file([LIB_NAME, mode], "median filter", median_filter_results)

drv.Context.pop()
