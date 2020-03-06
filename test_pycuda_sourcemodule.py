from pycuda.compiler import SourceModule
import numpy as np

from imagingtester import DTYPE, create_arrays
from pycuda_test_utils import PyCudaImplementation

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

    float neighb_array[20];

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

        if (0)
        {
            // find_median(neighb_array, filter_height * filter_width);
            for (int i = 0; i < filter_width * filter_height; i++)
                printf("%f ", neighb_array[i]);
        }

        data_array[(id_img * img_size) + (id_x * X) + id_y] = find_median(neighb_array, filter_height * filter_width);
    }
}
"""
)

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


test = PyCudaSourceModuleImplementation((20, 20, 20), DTYPE)
test.warm_up()
