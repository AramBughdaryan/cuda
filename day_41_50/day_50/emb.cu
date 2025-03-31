#include <cuda.h>
#include <iostream>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


__global__ void embedding_lookup_kernel(
    float* output,           // Output embeddings
    const int* indices,        // Input indices to look up
    const float* embedding_table,
    int vocab_size,           // Total vocabulary size
    int embedding_dim,        // Dimension of each embedding
    int batch_size            // Number of indices to look up
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
            // Optimization: Use vectorized copy if embedding_dim is divisible by 4 (or 2)
            // For simplicity, keeping the loop for now. Consider using cudaMemcpyDeviceToDevice for larger chunks
            // if performance is critical and indices are somewhat contiguous, although lookup is inherently sparse.
            for (int j = 0; j < embedding_dim; ++j) {
                output_start[j] = embedding_start[j];
            }
        } else {
             // Handle out-of-bounds indices if needed (e.g., zero out the output)
             // float* output_start = output + (idx * embedding_dim);
             // for (int j = 0; j < embedding_dim; ++j) {
             //     output_start[j] = 0.0f;
             // }
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

            // Accumulate gradients (using atomicAdd for thread safety, as multiple threads might write to the same embedding gradient)
            for (int j = 0; j < embedding_dim; ++j) {
                atomicAdd(&grad_target[j], grad_src[j]);
            }
        }
        // Note: No gradient contribution for out-of-bounds indices
    }
}


class EmbeddingFunction : public torch::autograd::Function<EmbeddingFunction> {
    public:
        static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            const torch::Tensor& indices,
            const torch::Tensor& embedding_table
        ) {
            // Ensure inputs are on CUDA and correct type
            TORCH_CHECK(indices.is_cuda(), "Indices tensor must be a CUDA tensor");
            // Ensure indices are integer type (long or int)
            TORCH_CHECK(indices.scalar_type() == torch::kInt32 || indices.scalar_type() == torch::kInt64, "Indices tensor must be int32 or int64");
            TORCH_CHECK(embedding_table.is_cuda(), "Embedding table must be a CUDA tensor");
            TORCH_CHECK(embedding_table.scalar_type() == torch::kFloat32, "Embedding table must be float32");

            // Get dimensions
            const int batch_size = indices.numel(); // Use numel() for flattened or multi-dim indices
            const int vocab_size = embedding_table.size(0);
            const int embedding_dim = embedding_table.size(1);

            // Create output tensor on the same device as inputs
            auto output = torch::empty({batch_size, embedding_dim},
                                       embedding_table.options()); // options() copies dtype, device, layout

            // Ensure indices tensor is contiguous and int32 for the kernel
            // Note: Kernels often expect int32. If indices are int64, cast them.
            // Make a contiguous copy if necessary.
            auto indices_contiguous = indices.contiguous();
            torch::Tensor indices_int32;
             if (indices_contiguous.scalar_type() == torch::kInt64) {
                 // Cast to int32 if needed - be careful about potential overflow if indices > 2^31-1
                 indices_int32 = indices_contiguous.to(torch::kInt32);
             } else {
                 indices_int32 = indices_contiguous;
             }


            // Compute grid and block dimensions
            const int blockSize = 256; // Common block size, can be tuned
            const int numBlocks = (batch_size + blockSize - 1) / blockSize;

            // Get the current CUDA stream
            // Use ATen's interface for better integration with PyTorch's stream management
            cudaStream_t stream = at::cuda::getCurrentCUDAStream();

            // Launch CUDA kernel
            embedding_lookup_kernel<<<numBlocks, blockSize, 0, stream>>>(
                output.data_ptr<float>(),
                indices_int32.data_ptr<int>(), // Use the int32 tensor
                embedding_table.data_ptr<float>(),
                vocab_size,
                embedding_dim,
                batch_size
            );
            // Check for kernel launch errors (useful for debugging)
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // Save tensors needed for backward. Save the int32 version if casted.
            ctx->save_for_backward({indices_int32, embedding_table});

            return output;
        }

        static std::vector<torch::Tensor> backward(
            torch::autograd::AutogradContext* ctx,
            std::vector<torch::Tensor> grad_outputs // grad_outputs is a vector, usually size 1
        ) {
            // Check if gradient output is provided
            if (grad_outputs.empty() || !grad_outputs[0].defined()) {
                 // Handle cases where no gradient needs to be computed back.
                 // Return undefined tensors for inputs that required grad.
                 // embedding_table requires grad, indices does not.
                 return {torch::Tensor(), torch::Tensor()};
            }

            auto saved = ctx->get_saved_variables();
            auto indices = saved[0]; // This is the potentially casted int32 tensor
            auto embedding_table = saved[1];

            const int batch_size = indices.numel();
            const int vocab_size = embedding_table.size(0);
            const int embedding_dim = embedding_table.size(1);

            // Ensure grad_output is contiguous, float32 and on CUDA
            auto grad_output = grad_outputs[0];
             TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
             TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "grad_output must be float32");
             // Check dimensions match expected output shape
             TORCH_CHECK(grad_output.size(0) == batch_size && grad_output.size(1) == embedding_dim, "grad_output shape mismatch");


            // Initialize gradient for embedding table with zeros. Must be on the same device.
            // Use zeros_like to ensure same dtype, device, layout.
            auto grad_embedding = torch::zeros_like(embedding_table);

            // Launch a custom kernel to handle gradients
            const int blockSize = 256;
            const int numBlocks = (batch_size + blockSize - 1) / blockSize;

            // Get the current CUDA stream
            cudaStream_t stream = at::cuda::getCurrentCUDAStream();

            embedding_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
                grad_output.data_ptr<float>(),
                indices.data_ptr<int>(), // indices were saved as int32
                grad_embedding.data_ptr<float>(),
                vocab_size,
                embedding_dim,
                batch_size
            );
             // Check for kernel launch errors
             C10_CUDA_KERNEL_LAUNCH_CHECK();

            // Return gradients for the inputs of the forward function:
            // First is gradient w.r.t. 'indices' (doesn't require grad, return undefined tensor)
            // Second is gradient w.r.t. 'embedding_table' (our computed grad_embedding)
            return {torch::Tensor(), grad_embedding};
        }
};

// torch::nn::Module subclass for the C++ part
class CudaEmbeddingImpl : public torch::nn::Module {
public:
    // Constructor
    CudaEmbeddingImpl(int64_t vocab_size, int64_t embedding_dim) {
        // Define tensor options: float32, requires_grad, CUDA device
        auto options = torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .layout(torch::kStrided) // Default layout
                           .device(torch::kCUDA)    // Specify CUDA device directly
                           .requires_grad(true);    // Set requires_grad

        // Create and register the weight parameter directly on CUDA
        weight = register_parameter("weight", torch::empty({vocab_size, embedding_dim}, options));

        // Initialize weights using torch::nn::init namespace (in-place)
        // No need to move to CUDA again, it's already there.
        torch::nn::init::normal_(weight, /*mean=*/0.0, /*std=*/0.02);
    }

    // Forward method
    torch::Tensor forward(const torch::Tensor& indices) {
        // Ensure indices are moved to CUDA if they aren't already
        // It's generally better practice to ensure input tensors are on the correct
        // device *before* calling the module's forward method, but adding a check
        // here can make the module more robust.
        auto indices_cuda = indices.device().is_cuda() ? indices : indices.to(torch::kCUDA);

        // Check if indices need to be cast to int32/int64 (kernel expects int)
        // The custom function handles the int32 check/cast
        // if (indices_cuda.scalar_type() != torch::kInt32 && indices_cuda.scalar_type() != torch::kInt64) {
        //     indices_cuda = indices_cuda.to(torch::kInt64); // Cast to long as standard nn.Embedding expects
        // }


        // Apply our custom embedding function
        return EmbeddingFunction::apply(indices_cuda, weight);
    }

    // Expose the weight parameter (already public via register_parameter, but explicit member is fine)
    torch::Tensor weight;

    // Note: The default zero_grad() from torch::nn::Module should work correctly now
    // as 'weight' is properly registered. You typically don't need to override it
    // unless you have non-parameter tensors that need grads zeroed.
    // void zero_grad() {
    //     if (weight.grad().defined()) {
    //         weight.grad().zero_();
    //     }
    // }
};


// PYBIND11 bindings (no changes needed here usually)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Bind the CudaEmbeddingImpl class, inheriting from torch::nn::Module
    py::class_<CudaEmbeddingImpl, std::shared_ptr<CudaEmbeddingImpl>, torch::nn::Module>(m, "CudaEmbedding")
        .def(py::init<int64_t, int64_t>(), py::arg("vocab_size"), py::arg("embedding_dim")) // Constructor
        .def("forward", &CudaEmbeddingImpl::forward, py::arg("indices")) // Forward method
        // Make the module callable (equivalent to forward)
        // If using __call__ like python, you might need to handle device placement more carefully inside __call__
        // or ensure inputs are correct device before calling. Usually binding 'forward' is sufficient.
        .def("__call__", &CudaEmbeddingImpl::forward)
        // Expose the 'weight' parameter so it can be accessed/modified from Python
        .def_readwrite("weight", &CudaEmbeddingImpl::weight);
        // .def("zero_grad", &CudaEmbeddingImpl::zero_grad); // Can likely remove this override
}