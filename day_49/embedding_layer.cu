#include <cuda.h>
#include <iostream>
#include <torch/extension.h>


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


__global__ void embedding_backward_kernel(
    const float* grad_output,  // Gradient from upstream [batch_size, embedding_dim]
    const int* indices,        // Input indices [batch_size]
    float* grad_embedding,     // Gradient for embedding table [vocab_size, embedding_dim]
    int vocab_size,
    int embedding_dim,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size) {
        // Get the vocabulary index for this sample
        int vocab_idx = indices[idx];
        
        // Bounds checking
        if (vocab_idx >= 0 && vocab_idx < vocab_size) {
            // Source gradient for this sample
            const float* grad_src = grad_output + (idx * embedding_dim);
            
            // Target location in the embedding gradient table
            float* grad_target = grad_embedding + (vocab_idx * embedding_dim);
            
            // Accumulate gradients (using atomicAdd for thread safety)
            for (int j = 0; j < embedding_dim; ++j) {
                atomicAdd(&grad_target[j], grad_src[j]);
            }
        }
    }
}


class EmbeddingFunction : public torch::autograd::Function<EmbeddingFunction> {
    public:
        static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            const torch::Tensor& indices,
            const torch::Tensor& embedding_table
        ) {
            // Ensure inputs are on CUDA
            TORCH_CHECK(indices.is_cuda(), "Indices tensor must be a CUDA tensor");
            TORCH_CHECK(embedding_table.is_cuda(), "Embedding table must be a CUDA tensor");
            
            // Get dimensions
            const int batch_size = indices.numel();
            const int vocab_size = embedding_table.size(0);
            const int embedding_dim = embedding_table.size(1);
            
            // Create output tensor
            auto output = torch::empty({batch_size, embedding_dim}, 
                                       embedding_table.options());
            
            // Compute grid and block dimensions
            const int blockSize = 256;
            const int numBlocks = (batch_size + blockSize - 1) / blockSize;
            
            // Launch CUDA kernel
            embedding_lookup_kernel<<<numBlocks, blockSize>>>(
                output.data_ptr<float>(),
                indices.data_ptr<int>(),
                embedding_table.data_ptr<float>(),
                vocab_size,
                embedding_dim,
                batch_size
            );
            
            // Save variables for backward
            ctx->save_for_backward({indices, embedding_table});
            
            return output;
        }
        
        static std::vector<torch::Tensor> backward(
            torch::autograd::AutogradContext* ctx,
            std::vector<torch::Tensor> grad_outputs
        ) {
            auto saved = ctx->get_saved_variables();
            auto indices = saved[0];
            auto embedding_table = saved[1];
            
            const int batch_size = indices.numel();
            const int vocab_size = embedding_table.size(0);
            const int embedding_dim = embedding_table.size(1);
            
            // Ensure grad_output is contiguous
            // TODO CHECK IF THERE IS NEED TO FOR OPERATION!(making contiguous)
            auto grad_output = grad_outputs[0].contiguous();
            
            // Initialize gradient for embedding table with zeros
            auto grad_embedding = torch::zeros_like(embedding_table);
            
            // Launch a custom kernel to handle gradients
            const int blockSize = 256;
            const int numBlocks = (batch_size + blockSize - 1) / blockSize;
            
            embedding_backward_kernel<<<numBlocks, blockSize>>>(
                grad_output.data_ptr<float>(),
                indices.data_ptr<int>(),
                grad_embedding.data_ptr<float>(),
                vocab_size,
                embedding_dim,
                batch_size
            );

            return {torch::Tensor(), grad_embedding};
        }
};

class CudaEmbedding : public torch::nn::Module {
public:
    CudaEmbedding(int64_t vocab_size, int64_t embedding_dim) {
        embedding_table = register_parameter("weight",
             torch::empty({vocab_size, embedding_dim})
            );
        
        torch::nn::init::normal_(embedding_table, 0.0, 0.02);
        embedding_table = embedding_table.to(torch::kCUDA);
    }

    torch::Tensor forward(const torch::Tensor& indices) {
        return EmbeddingFunction::apply(indices, embedding_table);
    }

    torch::Tensor weight() { return embedding_table; }
    
    void set_weight(const torch::Tensor& new_weight) {
        TORCH_CHECK(new_weight.sizes() == embedding_table.sizes(), 
                   "New weight tensor must have the same dimensions as the embedding table");
        embedding_table.data().copy_(new_weight);
    }
    void zero_grad() {
        embedding_table.grad().zero_();
    }
};


// Benchmark function
void benchmark_embedding(int vocab_size, int embedding_dim, int batch_size, int num_runs) {
    // Create test inputs
    auto indices = torch::randint(0, vocab_size, {batch_size}, torch::kInt32).to(torch::kCUDA);
    
    // Create our custom embedding layer
    auto custom_embedding = std::make_shared<CudaEmbedding>(vocab_size, embedding_dim);
    custom_embedding->to(torch::kCUDA);
    
    // Create PyTorch's embedding layer
    auto pytorch_embedding = torch::nn::Embedding(vocab_size, embedding_dim);
    pytorch_embedding->to(torch::kCUDA);
    
    // Make sure weights are identical for correctness comparison
    pytorch_embedding->weight.data().copy_(custom_embedding->weight());
    
    // Create random upstream gradients for backward pass
    auto upstream_grad = torch::randn({batch_size, embedding_dim}, torch::kCUDA).requires_grad_(true);
    
    // Warmup runs
    for (int i = 0; i < 5; i++) {
        auto custom_out = custom_embedding->forward(indices);
        auto pytorch_out = pytorch_embedding->forward(indices);
    }
    
    // Benchmark forward pass
    std::cout << "Benchmarking forward pass...\n";
    
    // Custom embedding benchmark
    auto start_time = std::chrono::high_resolution_clock::now();
    torch::Tensor custom_result;
    for (int i = 0; i < num_runs; i++) {
        custom_result = custom_embedding->forward(indices);
        torch::cuda::synchronize();
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto custom_forward_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / num_runs;
    
    // PyTorch embedding benchmark
    start_time = std::chrono::high_resolution_clock::now();
    torch::Tensor pytorch_result;
    for (int i = 0; i < num_runs; i++) {
        pytorch_result = pytorch_embedding->forward(indices);
        torch::cuda::synchronize();
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto pytorch_forward_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / num_runs;
    
    // Benchmark backward pass
    std::cout << "Benchmarking backward pass...\n";
    
    // Custom embedding backward benchmark
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        auto out = custom_embedding->forward(indices);
        auto loss = out.mul(upstream_grad).sum();
        loss.backward();
        torch::cuda::synchronize();
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto custom_backward_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / num_runs;
    
    // Get custom gradients for comparison
    auto custom_grads = custom_embedding->weight().grad().clone();
    custom_embedding->zero_grad();
    
    // PyTorch embedding backward benchmark
    start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        auto out = pytorch_embedding->forward(indices);
        auto loss = out.mul(upstream_grad).sum();
        loss.backward();
        torch::cuda::synchronize();
    }
    end_time = std::chrono::high_resolution_clock::now();
    auto pytorch_backward_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / num_runs;
    
    // Get PyTorch gradients for comparison
    auto pytorch_grads = pytorch_embedding->weight.grad().clone();
    pytorch_embedding->zero_grad();
    
    // Print results
    std::cout << "==== Benchmark Results ====\n";
    std::cout << "Configuration: vocab_size=" << vocab_size << ", embedding_dim=" << embedding_dim 
              << ", batch_size=" << batch_size << ", runs=" << num_runs << "\n\n";
    
    std::cout << "Forward Pass:\n";
    std::cout << "Custom implementation: " << custom_forward_time << " μs\n";
    std::cout << "PyTorch implementation: " << pytorch_forward_time << " μs\n";
    std::cout << "Speedup: " << (float)pytorch_forward_time / custom_forward_time << "x\n\n";
    
    std::cout << "Backward Pass:\n";
    std::cout << "Custom implementation: " << custom_backward_time << " μs\n";
    std::cout << "PyTorch implementation: " << pytorch_backward_time << " μs\n";
    std::cout << "Speedup: " << (float)pytorch_backward_time / custom_backward_time << "x\n\n";
    
    // Check correctness
    std::cout << "==== Correctness Check ====\n";
    
    // Forward pass check
    auto forward_diff = (custom_result - pytorch_result).abs().max().item<float>();
    std::cout << "Forward pass maximum absolute difference: " << forward_diff << "\n";
    bool forward_match = forward_diff < 1e-5;
    std::cout << "Forward pass match: " << (forward_match ? "YES" : "NO") << "\n\n";
    
    // Backward pass check
    auto backward_diff = (custom_grads - pytorch_grads).abs().max().item<float>();
    std::cout << "Backward pass maximum absolute difference: " << backward_diff << "\n";
    bool backward_match = backward_diff < 1e-5;
    std::cout << "Backward pass match: " << (backward_match ? "YES" : "NO") << "\n";
}


torch::Tensor cuda_embedding_forward(
    const torch::Tensor& indices,
    const torch::Tensor& embedding_table
) {
    return EmbeddingFunction::apply(indices, embedding_table);
}

// Create the Python module
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     // m.def("embedding_forward", &cuda_embedding_forward, "Custom CUDA embedding forward function");
//     m.def("EmbeddingLayer", [](int64_t vocab_size, int64_t embedding_dim) {
//         return std::make_shared<CudaEmbedding>(vocab_size, embedding_dim);
//     }, "Custom CUDA Embedding Layer");
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<CudaEmbedding, std::shared_ptr<CudaEmbedding>>(m, "CudaEmbedding")
        .def(py::init<int64_t, int64_t>())
        .def("forward", &CudaEmbedding::forward)
        .def("weight", &CudaEmbedding::weight)
        .def("set_weight", &CudaEmbedding::set_weight)
        .def("zero_grad", [](CudaEmbedding &self) { self.zero_grad(); });
}

int main() {
    // Check if CUDA is available
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available! Exiting...\n";
        return 1;
    }
    
    std::cout << "Running embedding layer benchmark...\n";
    
    // Small config
    std::cout << "\n=== Small Configuration ===\n";
    benchmark_embedding(1000, 64, 128, 100);
    
    // Medium config
    std::cout << "\n=== Medium Configuration ===\n";
    benchmark_embedding(10000, 256, 512, 50);
    
    // Large config
    std::cout << "\n=== Large Configuration ===\n";
    benchmark_embedding(50000, 512, 1024, 20);
    
    return 0;
}