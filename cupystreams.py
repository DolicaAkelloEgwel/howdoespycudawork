import numpy as np
import cupy as cp

array_size = (1000, 1000, 1000)


def pinned_array(array):
    # first constructing pinned memory
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src


a_cpu = np.ones(array_size, dtype=np.float32)

# np.ndarray with pinned memory
a_cpu = pinned_array(a_cpu)

a_stream = cp.cuda.Stream(non_blocking=True)

a_gpu = cp.empty(a_cpu.shape, dtype="float32")

a_gpu.set(a_cpu, stream=a_stream)

# wait until a_cpu is copied in a_gpu
a_stream.synchronize()

# This line runs parallel to b_gpu.set()
a_gpu *= 2
