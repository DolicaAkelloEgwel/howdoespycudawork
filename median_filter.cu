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
//                    neighb_array[n_counter] = padded_array[(id_img * padded_img_size) + (padded_img_width * (filter_height / 2 + i)) + (filter_width / 2) + j];
                    neighb_array[n_counter] = padded_array[(id_img * padded_img_size) + (i * padded_img_width) + j];
                    n_counter += 1;
                }
            }
            data_array[(id_img * img_size) + (id_x * X) + id_y] = find_median(neighb_array, filter_height * filter_width);
        }
    }
}