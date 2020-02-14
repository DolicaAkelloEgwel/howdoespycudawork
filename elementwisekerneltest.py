import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit

from pycuda.elementwise import ElementwiseKernel

from imagingtester import create_arrays, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE

BackgroundCorrection = ElementwiseKernel(
    arguments="float * data, float * flat, const float * dark, const float MINIMUM_PIXEL_VALUE, const float MAXIMUM_PIXEL_VALUE",
    operation="flat[i] -= dark[i];"
    "if (flat[i] <= 0) flat[i] = MINIMUM_PIXEL_VALUE;"
    "data[i] -= dark[i];"
    "data[i] /= flat[i];"
    "if (flat[i] > MAXIMUM_PIXEL_VALUE) flat[i] = MAXIMUM_PIXEL_VALUE;"
    "if (flat[i] < MINIMUM_PIXEL_VALUE) flat[i] = MINIMUM_PIXEL_VALUE;",
    name="BackgroundCorrection",
)
# warm-up
warm_up_size = (1, 1, 1)
arrays = create_arrays(warm_up_size, "float32")
print(arrays[0])
arrays = [gpuarray.to_gpu(array) for array in arrays]
BackgroundCorrection(
    arrays[0], arrays[1], arrays[2], MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
)

# real deal
challenging_size = (1000, 1000, 500)
arrays = create_arrays(challenging_size, "float32")
start = drv.Event()
end = drv.Event()
start.record()
arrays = [gpuarray.to_gpu(array) for array in arrays]
BackgroundCorrection(
    arrays[0], arrays[1], arrays[2], MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
)
end.record()
end.synchronize()
secs = start.time_till(end) / 1e3
print(secs)
