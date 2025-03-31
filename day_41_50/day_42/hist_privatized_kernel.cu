#include <iostream>
#include <cuda.h>

__global__ void histogram_privatized_kernel(unsigned char* input, usigned char* bins, 
            unsigned int num_elements, unsigned int num_bins) {
        
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

        extern __shared__ unsigned int histo_s[];
        for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
            histo_s[binIdx] = 0u;
        }
        __syncthreads();

        unsigned int prev_index = -1;
        unsigned int accumulator = 0;

        for(unsigned int = tid; i < num_elements; i += blockDim.x * gridDim.x) {
            int alphabet_position = input[i] - "a";
            if (alphabet_position >=0 && alphabet_position < 26) {
                unsigned int curr_index = alphabet_position / 4;
                if (curr_index != prev_index) {
                    if (accumulator >= 0 ) atomicAdd(&(histo_s[alphabet_position/4]), accumulator)
                    accumulator = 1;
                    prev_index = curr_index;
                }
                else {
                    accumulator++;
                }
            }
        }
        __syncthreads();

        for (unsigned int binIdx = threadIdx.x; binIdx < num_bins; binIdx += blockDim.x) {
            atomicAdd(&(bins[binIdx]), histo_s[binIdx]);
        }
    }
