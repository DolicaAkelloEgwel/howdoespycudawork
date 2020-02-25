from numba import cuda
import time
import numpy as np

from imagingtester import (
    MINIMUM_PIXEL_VALUE,
    MAXIMUM_PIXEL_VALUE,
    N_RUNS,
    DTYPE,
    create_arrays,
    SIZES_SUBSET,
    get_array_partition_indices,
    PRINT_INFO,
    num_partitions_needed,
)
from numba_test_utils import (
    create_vectorise_add_arrays,
    create_vectorise_background_correction,
    LIB_NAME,
    get_free_bytes,
    NumbaImplementation,
)
from numpy_background_correction import numpy_background_correction
from write_and_read_results import (
    write_results_to_file,
    ADD_ARRAYS,
    BACKGROUND_CORRECTION,
    ARRAY_SIZES,
)

mode = "vectorise"


add_arrays = create_vectorise_add_arrays("cuda")
background_correction = create_vectorise_background_correction("cuda")


class NumbaCudaVectoriseImplementation(NumbaImplementation):
    def __init__(self, size, dtype):
        super().__init__(size, dtype)

    def warm_up(self):
        """
        Give the CUDA functions a chance to compile.
        """
        warm_up_arrays = create_arrays((1, 1, 1), self.dtype)
        add_arrays(*warm_up_arrays[:2])
        background_correction(*warm_up_arrays, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE)

    def get_time(self):
        self.synchronise()
        return time.time()

    def timed_imaging_operation(
        self, runs, alg, alg_name, n_arrs_needed, n_gpu_arrs_needed
    ):

        # Synchronize and free memory before making an assessment about available space
        self.clear_cuda_memory()

        n_partitions_needed = num_partitions_needed(
            self.cpu_arrays[0], n_gpu_arrs_needed, get_free_bytes()
        )

        transfer_time = 0
        operation_time = 0

        if n_partitions_needed == 1:

            # Time transfer from CPU to GPU
            start = self.get_time()
            gpu_arrays = self._send_arrays_to_gpu(
                self.cpu_arrays[:n_arrs_needed], n_gpu_arrs_needed
            )
            transfer_time += self.get_time() - start

            # Repeat the operation
            for _ in range(runs):
                operation_time += self.time_function(
                    lambda: alg(*gpu_arrays[:n_arrs_needed])
                )

            # Free the GPU arrays
            self.clear_cuda_memory(gpu_arrays)

        else:

            # Determine the number of partitions required again (to be on the safe side)
            n_partitions_needed = num_partitions_needed(
                self.cpu_arrays[0], n_gpu_arrs_needed, get_free_bytes()
            )

            indices = get_array_partition_indices(
                self.cpu_arrays[0].shape[0], n_partitions_needed
            )

            for i in range(n_partitions_needed):

                # Retrieve the segments used for this iteration of the operation
                split_cpu_arrays = [
                    cpu_array[indices[i][0] : indices[i][1] :, :]
                    for cpu_array in self.cpu_arrays
                ]

                # Time transferring the segments to the GPU
                start = self.get_time()
                gpu_arrays = self._send_arrays_to_gpu(
                    split_cpu_arrays, n_gpu_arrs_needed
                )
                transfer_time += self.get_time() - start

                if not gpu_arrays:
                    return 0

                # Carry out the operation on the slices
                for _ in range(runs):
                    operation_time += self.time_function(
                        lambda: alg(*gpu_arrays[:n_arrs_needed])
                    )

                # Free GPU arrays and partition arrays
                self.clear_cuda_memory(gpu_arrays)

        if transfer_time > 0 and operation_time > 0:
            self.print_operation_times(operation_time, alg_name, runs, transfer_time)

        return transfer_time + operation_time / runs


practice_array = np.ones(shape=(5, 5, 5)).astype(DTYPE)
vect_result = add_arrays(practice_array, practice_array)
assert np.all(vect_result == 2)

# Checking the two background corrections get the same result
np_data, np_dark, np_flat = [
    np.random.uniform(low=0.0, high=20, size=(5, 5, 5)) for _ in range(3)
]

numba_data = np_data.copy()
numba_dark = np_dark.copy()
numba_flat = np_flat.copy()

background_correction(
    numba_data, numba_dark, numba_flat, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
)
numpy_background_correction(
    np_data, np_dark, np_flat, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
)
assert np.allclose(np_data, numba_data)

add_arrays_results = []
background_correction_results = []


def background_correction_fixed_clip(dark, data, flat):
    return background_correction(
        data, dark, flat, MINIMUM_PIXEL_VALUE, MAXIMUM_PIXEL_VALUE
    )


for size in ARRAY_SIZES[:SIZES_SUBSET]:

    imaging_obj = NumbaCudaVectoriseImplementation(size, DTYPE)

    try:
        avg_add = imaging_obj.timed_imaging_operation(
            N_RUNS, add_arrays, "adding", 2, 2
        )
        avg_bc = imaging_obj.timed_imaging_operation(
            N_RUNS, background_correction_fixed_clip, "background correction", 3, 4
        )

        if avg_add > 0:
            add_arrays_results.append(avg_add)
        if avg_bc > 0:
            background_correction_results.append(avg_bc)

    except cuda.cudadrv.driver.CudaAPIError:
        if PRINT_INFO:
            print("Can't operate on arrays with size:", size)
            print("Free bytes during CUDA error:", get_free_bytes())
        imaging_obj.clear_cuda_memory()
        break

write_results_to_file([LIB_NAME, mode], ADD_ARRAYS, add_arrays_results)
write_results_to_file(
    [LIB_NAME, mode], BACKGROUND_CORRECTION, background_correction_results
)
