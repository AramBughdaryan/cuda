#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

const int VOCAB_SIZE = 10000;      // Total unique tokens
const int EMBEDDING_DIM = 128;     // Size of each embedding vector
const int BATCH_SIZE = 1024;       // Number of indices to process
const int NUM_ITERATIONS = 10;   // Benchmark iterations


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
    }
}

__global__ void embedding_lookup_texture_kernel(
    float* output,           // Output embeddings
    const int* indices,      // Input indices to look up
    cudaTextureObject_t embedding_texture, // Texture object for embedding table
    int vocab_size,          // Total vocabulary size
    int embedding_dim,       // Dimension of each embedding
    int batch_size           // Number of indices to look up
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < batch_size) {
        int vocab_index = indices[idx];

        if (vocab_index >= 0 && vocab_index < vocab_size) {
            float* output_start = output + (idx * embedding_dim);

            for (int j = 0; j < embedding_dim; ++j) {
                output_start[j] = tex2D<float>(embedding_texture, j, vocab_index);
            }
        }
    }
}

// Host function to launch embedding lookup
void embedding_lookup(
    float* d_output,
    const int* d_indices,
    const float* d_embedding_table,
    int vocab_size,
    int embedding_dim,
    int batch_size
) {
    // Calculate grid and block dimensions
    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    embedding_lookup_kernel<<<blocks, threads_per_block>>>(
        d_output, 
        d_indices, 
        d_embedding_table, 
        vocab_size, 
        embedding_dim, 
        batch_size
    );
}


cudaTextureObject_t prepare_embedding_texture(float* d_embedding_table, int vocab_size, int embedding_dim) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, embedding_dim, vocab_size);

    cudaMemcpyToArray(cuArray, 0, 0, d_embedding_table,
                    vocab_size * embedding_dim * sizeof(float),
                    cudaMemcpyDeviceToDevice);
    
    cudaTextureObject_t tex_obj;
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL);

    return tex_obj;
}

void benchmark_embedding_lookup() {
    // Device Memory Allocation
    int* d_indices;
    float* d_embedding_table;
    float* d_output_standard;
    float* d_output_texture;

    // Memory Allocation
    CUDA_CHECK(cudaMalloc(&d_indices, BATCH_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_embedding_table, VOCAB_SIZE * EMBEDDING_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_standard, BATCH_SIZE * EMBEDDING_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_texture, BATCH_SIZE * EMBEDDING_DIM * sizeof(float)));

    // Host Memory for Indices
    int* h_indices = new int[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; ++i) {
        h_indices[i] = rand() % VOCAB_SIZE;
    }

    // Allocate and initialize embedding table
    float* h_embedding_table = new float[VOCAB_SIZE * EMBEDDING_DIM];
    for (int i = 0; i < VOCAB_SIZE * EMBEDDING_DIM; ++i) {
        h_embedding_table[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(
        d_indices, h_indices, 
        BATCH_SIZE * sizeof(int), 
        cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_embedding_table, h_embedding_table, 
        VOCAB_SIZE * EMBEDDING_DIM * sizeof(float), 
        cudaMemcpyHostToDevice
    ));

    // Prepare Texture Object
    cudaTextureObject_t tex_embedding = prepare_embedding_texture(
        d_embedding_table, 
        VOCAB_SIZE, 
        EMBEDDING_DIM
    );

    // Kernel Launch Configuration
    int threads_per_block = 256;
    int blocks = (BATCH_SIZE + threads_per_block - 1) / threads_per_block;

    // Standard Embedding Timing
    auto start_standard = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        embedding_lookup_kernel<<<blocks, threads_per_block>>>(
            d_output_standard, 
            d_indices, 
            d_embedding_table, 
            VOCAB_SIZE, 
            EMBEDDING_DIM, 
            BATCH_SIZE
        );
    }
    cudaDeviceSynchronize();
    auto end_standard = std::chrono::high_resolution_clock::now();

    // Texture Embedding Timing
    auto start_texture = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        embedding_lookup_texture_kernel<<<blocks, threads_per_block>>>(
            d_output_texture, 
            d_indices, 
            tex_embedding, 
            VOCAB_SIZE, 
            EMBEDDING_DIM, 
            BATCH_SIZE
        );
    }
    cudaDeviceSynchronize();
    auto end_texture = std::chrono::high_resolution_clock::now();

    // Calculate Durations
    auto standard_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_standard - start_standard
    );
    auto texture_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_texture - start_texture
    );

    // Performance Output
    std::cout << "Benchmark Results:\n";
    std::cout << "Standard Embedding Lookup: " 
              << standard_duration.count() << " microseconds\n";
    std::cout << "Texture Embedding Lookup: " 
              << texture_duration.count() << " microseconds\n";
    std::cout << "Speedup: " 
              << static_cast<float>(standard_duration.count()) / texture_duration.count() 
              << "x\n";

    // Verify Results (Optional)
    float* h_output_standard = new float[BATCH_SIZE * EMBEDDING_DIM];
    float* h_output_texture = new float[BATCH_SIZE * EMBEDDING_DIM];

    CUDA_CHECK(cudaMemcpy(
        h_output_standard, d_output_standard, 
        BATCH_SIZE * EMBEDDING_DIM * sizeof(float), 
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        h_output_texture, d_output_texture, 
        BATCH_SIZE * EMBEDDING_DIM * sizeof(float), 
        cudaMemcpyDeviceToHost
    ));

    // Simple verification
    bool results_match = true;
    for (int i = 0; i < BATCH_SIZE * EMBEDDING_DIM; ++i) {
        if (std::abs(h_output_standard[i] - h_output_texture[i]) > 1e-5) {
            results_match = false;
            break;
        }
    }

    std::cout << "Results Match: " << (results_match ? "YES" : "NO") << std::endl;

    // Cleanup
    delete[] h_indices;
    delete[] h_embedding_table;
    delete[] h_output_standard;
    delete[] h_output_texture;
    cudaFree(d_indices);
    cudaFree(d_embedding_table);
    cudaFree(d_output_standard);
    cudaFree(d_output_texture);
}

int main() {
    benchmark_embedding_lookup();
    return 0;
}
