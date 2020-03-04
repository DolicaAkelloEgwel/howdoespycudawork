extern "C"{
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
        return neighb_array[N / 2 + 1];
    }
    __global__ void median_filter(float*** data_array, const float*** padded_array, const int N_IMAGES, const int X, const int Y, const int pad_height, const int pad_width, const int filter_width, const int neighb_size)
    {
        unsigned int id_img = blockIdx.x*blockDim.x + threadIdx.x;
        unsigned int id_x = blockIdx.y*blockDim.y + threadIdx.y;
        unsigned int id_y = blockIdx.z*blockDim.z + threadIdx.z;

        if ((id_img > N_IMAGES) || (id_x > X) || (id_y > Y))
            return;

        float neighb_array[20];
//        cudaMalloc((void**)&neighb_array, neighb_size * sizeof(float));

        for (int i = id_x - pad_height; i < id_x + pad_height; i++)
        {
            for (int j = id_y - pad_width; j < id_y + pad_width; j++)
            {
                neighb_array[i * filter_width + j] = padded_array[id_img][id_x][id_y];
            }
        }

        data_array[id_img][id_x][id_y] = find_median(neighb_array, neighb_size);
//        cudaFree(neighb_array);
    }
}