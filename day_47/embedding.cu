#include <cuda_runtime.h>
#include <iostream>

__global__ void embedding_lookup_kernel(
    float* output,           // Output embeddings
    const int* indices,      // Input indices to look up
    const float* embedding_table, 
    int vocab_size,          // Total vocabulary size
    int embedding_dim,       // Dimension of each embedding
    int batch_size           // Number of indices to look up
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within batch size
    if (idx < batch_size) {
        // Get the vocabulary index for this sample
        int vocab_idx = indices[idx];
        
        // Bounds checking
        if (vocab_idx >= 0 && vocab_idx < vocab_size) {
            // Calculate start of embedding for this index
            const float* embedding_start = embedding_table + (vocab_idx * embedding_dim);
            
            // Calculate output start for this sample
            float* output_start = output + (idx * embedding_dim);
            
            // Copy embedding to output
            for (int j = 0; j < embedding_dim; ++j) {
                output_start[j] = embedding_start[j];
            }
        }
        else {
            // Handle out-of-bounds index (optional: zero out or use a default embedding)
            for (int j = 0; j < embedding_dim; ++j) {
                output[idx * embedding_dim + j] = 0.0f;
            }
        }
    }
}