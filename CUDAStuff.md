### Resources

[CUDA C Best Practices](https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)

### Manipulating 3D Arrays

```C
unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;
unsigned idy = blockIdx.y*blockDim.y + threadIdx.y;
unsigned idz = blockIdx.z*blockDim.z + threadIdx.z;

if ((idx < (DATAXSIZE)) && (idy < (DATAYSIZE)) && (idz < (DATAZSIZE)))
{
    a[idz][idy][idx] = idz+idy+idx;
}
```
### Optimising Data Transfer

Pinned memory vs. zero-copy memory.

### Grid Size and Block Size

???

