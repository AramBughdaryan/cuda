#include <iostream>
#include <cuda.h>

__global__ void histogram_privatized_kernel(usigned char* input, usinged int* bins,
                                    unsigned int num_elements, unsinged int num_bins) {
        
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ unsigned int histo_s[];

    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        histo_s[i] = 0u;
    }
    __syncthreads();

    for (unsigned int i = tid; i < num_elements; i += blockDim.x * gridDim.x) {
        int alphabet_position = buffer[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo_s[alphabet_position/4], 1);
        }
    }
    __syncthreads();
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&bins[i], histo_s[i]);
    }
}
