from pycuda import gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np

from imagingtester import create_arrays, DTYPE, SIZES_SUBSET
from numpy_background_correction import numpy_background_correction
from pycuda_test_utils import PyCudaImplementation, C_DTYPE, LIB_NAME

from write_and_read_results import (
    ARRAY_SIZES,
    write_results_to_file,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
)

BLOCKSIZE = 32


def iDivUp(a, b):
    return a // b + 1


blockDim = (BLOCKSIZE, 1, 1)


mode = "sourcemodule"

kernel_code = """
// the kernel definition
  __global__ void kernel(float *data, float *odata) {
  int index = blockIdx.x * blockDim.x + threadIdx.x ;
  odata[index] = data[index]+1.0f;

  }
  """.format(
    C_DTYPE
)

add_kernel = """
        __global__ void Add({0} *arr1, {0} *arr2)
        {
          int i = blockIdx.x * blockDim.x + threadIdx.x ;
          int j = blockIdx.y * blockDim.y + threadIdx.y ;
          int k = blockIdx.z * blockDim.z + threadIdx.z ;
          arr1[i] += arr2[i];
        }
        """.format(
    C_DTYPE
)

mod = SourceModule(kernel_code)

AddArrays = mod.get_function("Add")


class PyCudaSourceModuleImplementation(PyCudaImplementation):
    def __init__(self, size, dtype):

        super().__init__(size, dtype)
        self.warm_up()

    def warm_up(self):
        warm_up_size = (1, 1, 1)
        cpu_arrays = create_arrays(warm_up_size, DTYPE)
        gpu_arrays = self._send_arrays_to_gpu(cpu_arrays, 3)
        # BackgroundCorrection(
        #     gpu_arrays[0],
        #     gpu_arrays[1],
        #     gpu_arrays[2],
        #     MINIMUM_PIXEL_VALUE,
        #     MAXIMUM_PIXEL_VALUE,
        # )
        AddArrays(gpu_arrays[0], gpu_arrays[1])


practice_array = np.ones(shape=(5, 5, 5)).astype(DTYPE)
practice_array = gpuarray.to_gpu(practice_array)
# AddArrays(practice_array, practice_array)
assert np.all(practice_array.get() == 2)

np_arrays = [np.random.uniform(low=0.0, high=20, size=(5, 5, 5)) for _ in range(3)]
np_data, np_dark, np_flat = np_arrays
cuda_data, cuda_dark, cuda_flat = [gpuarray.to_gpu(np_arr) for np_arr in np_arrays]
# elementwise_background_correction(cuda_data, cuda_flat, cuda_dark)
numpy_background_correction(np_dark, np_data, np_flat)
assert np.allclose(np_data, cuda_data.get())

add_arrays_results = []
background_correction_results = []

for size in ARRAY_SIZES[:SIZES_SUBSET]:

    imaging_obj = PyCudaSourceModuleImplementation(size, DTYPE)

    # # add_arrays_results.append(
    #     imaging_obj.timed_imaging_operation(N_RUNS, AddArraysKernel, "adding", 2, 2)
    # # )
    # background_correction_results.append(
    #     imaging_obj.timed_imaging_operation(
    #         # N_RUNS, elementwise_background_correction, "background correction", 3, 3
    #     )
    # )

write_results_to_file([LIB_NAME, mode], ADD_ARRAYS, add_arrays_results)
write_results_to_file(
    [LIB_NAME, mode], BACKGROUND_CORRECTION, background_correction_results
)

drv.Context.pop()
