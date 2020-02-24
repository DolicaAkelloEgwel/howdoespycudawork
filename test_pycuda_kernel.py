from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
import numpy as np

from imagingtester import MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE, create_arrays, DTYPE
from numpy_background_correction import numpy_background_correction
from pycuda_test_utils import PyCudaImplementation, _send_arrays_to_gpu, C_DTYPE

# Create an element-wise Background Correction Function
BackgroundCorrectionKernel = ElementwiseKernel(
    arguments="{0} * data, {0} * flat, const {0} * dark, const {0} MINIMUM_PIXEL_VALUE, const {0} MAXIMUM_PIXEL_VALUE".format(
        C_DTYPE
    ),
    operation="flat[i] -= dark[i];"
    "if (flat[i] == 0) flat[i] = MINIMUM_PIXEL_VALUE;"
    "data[i] -= dark[i];"
    "data[i] /= flat[i];"
    "if (data[i] > MAXIMUM_PIXEL_VALUE) data[i] = MAXIMUM_PIXEL_VALUE;"
    "if (data[i] < MINIMUM_PIXEL_VALUE) data[i] = MINIMUM_PIXEL_VALUE;",
    name="BackgroundCorrectionKernel",
)

elementwise_background_correction = lambda data, flat, dark: BackgroundCorrectionKernel(
    data, flat, dark, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
)

# Create an element-wise Add Array Function
AddArraysKernel = ElementwiseKernel(
    arguments="{0} * arr1, {0} * arr2".format(C_DTYPE),
    operation="arr1[i] += arr2[i]",
    name="AddArraysKernel",
)


class PyCudaKernelImplementation(PyCudaImplementation):
    def __init__(self, size, dtype):

        super().__init__(size, dtype)
        self.warm_up()

    def warm_up(self):
        warm_up_size = (1, 1, 1)
        cpu_arrays = create_arrays(warm_up_size, DTYPE)
        gpu_arrays = _send_arrays_to_gpu(cpu_arrays, 3)
        BackgroundCorrectionKernel(
            gpu_arrays[0],
            gpu_arrays[1],
            gpu_arrays[2],
            MINIMUM_PIXEL_VALUE,
            MAXIMUM_PIXEL_VALUE,
        )
        AddArraysKernel(gpu_arrays[0], gpu_arrays[1], gpu_arrays[2])


practice_array = np.ones(shape=(5, 5, 5)).astype(DTYPE)
practice_array = gpuarray.to_gpu(practice_array)
AddArraysKernel(practice_array, practice_array)
assert np.all(practice_array.get() == 2)

np_arrays = [np.random.uniform(low=0.0, high=20, size=(5, 5, 5)) for _ in range(3)]
np_data, np_dark, np_flat = np_arrays
cuda_data, cuda_dark, cuda_flat = [gpuarray.to_gpu(np_arr) for np_arr in np_arrays]
elementwise_background_correction(cuda_data, cuda_flat, cuda_dark)
numpy_background_correction(np_dark, np_data, np_flat)
assert np.allclose(np_data, cuda_data.get())
