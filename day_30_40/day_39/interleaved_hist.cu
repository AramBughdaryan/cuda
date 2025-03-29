#include <iostream>
#include <cuda.h>

__global__ void histo_kernel(char *data, unsigned int length, unsigned int *histo) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int = tid; i < length; i += blockDim.x * gridDim.x) {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo[alphabet_position/4], 1);
        }
    }
}