#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat> // For FLT_MIN

// Simple cooperative reduction function (can be optimized further)
__device__ void blockReduceMax(float& value, float* shared_mem) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int lane = tid % warpSize; // Lane index within the warp (0-31)
    int warp = tid / warpSize; // Warp index within the block

    // Store thread's value into shared memory
    shared_mem[tid] = value;
    __syncthreads(); // Ensure all values are in shared memory

    // Reduce within the block (log_2 steps)
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_mem[tid] = max(shared_mem[tid], shared_mem[tid + offset]);
        }
        __syncthreads(); // Wait for all threads to complete the step
    }
    // After the loop, shared_mem[0] holds the block's maximum
    // Broadcast the max back to the thread's value variable
    value = shared_mem[0];
     __syncthreads(); // Ensure all threads read the correct max before proceeding
}
