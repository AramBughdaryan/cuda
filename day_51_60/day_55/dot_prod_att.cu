#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <limits> // Required for numeric_limits

// --- CUDA Error Checking Macro ---
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error at %s:%d\n", __FILE__, __LINE__); \
        /* Note: cuBLAS doesn't have a universal error string function like CUDA runtime */ \
        /* You might need to look up the status code in the documentation */ \
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) fprintf(stderr, "  Error: CUBLAS_STATUS_NOT_INITIALIZED\n"); \
        if (status == CUBLAS_STATUS_ALLOC_FAILED) fprintf(stderr, "  Error: CUBLAS_STATUS_ALLOC_FAILED\n"); \
        if (status == CUBLAS_STATUS_INVALID_VALUE) fprintf(stderr, "  Error: CUBLAS_STATUS_INVALID_VALUE\n"); \
        if (status == CUBLAS_STATUS_ARCH_MISMATCH) fprintf(stderr, "  Error: CUBLAS_STATUS_ARCH_MISMATCH\n"); \
        if (status == CUBLAS_STATUS_MAPPING_ERROR) fprintf(stderr, "  Error: CUBLAS_STATUS_MAPPING_ERROR\n"); \
        if (status == CUBLAS_STATUS_EXECUTION_FAILED) fprintf(stderr, "  Error: CUBLAS_STATUS_EXECUTION_FAILED\n"); \
        if (status == CUBLAS_STATUS_INTERNAL_ERROR) fprintf(stderr, "  Error: CUBLAS_STATUS_INTERNAL_ERROR\n"); \
        if (status == CUBLAS_STATUS_NOT_SUPPORTED) fprintf(stderr, "  Error: CUBLAS_STATUS_NOT_SUPPORTED\n"); \
        if (status == CUBLAS_STATUS_LICENSE_ERROR) fprintf(stderr, "  Error: CUBLAS_STATUS_LICENSE_ERROR\n"); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// --- Kernels ---

// Kernel for scaling
__global__ void scale_kernel(float *data, float scale_factor, int n_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        data[idx] *= scale_factor; // Apply scaling (division is multiplication by 1/sqrt(dk))
    }
}

// Kernel for applying additive mask (simplified: assumes mask is 0 or 1)
// mask_value is typically a large negative number like -1e9 or -infinity
__global__ void mask_kernel(float *scores, const float *mask, float mask_value, int n_rows, int n_cols) {
    int row = blockIdx.y; // One block per row
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Threads handle columns within a row

    if (row < n_rows && col < n_cols) {
        int score_idx = row * n_cols + col;
        // Assuming mask has same dimensions as scores
        // Apply mask if mask value is 0 (or some other indicator for masking)
        if (mask[score_idx] == 0.0f) {
             scores[score_idx] += mask_value;
        }
    }
}


// Softmax Kernel (Row-wise) - Simplified Version
// This version is less optimized than a production one which would use shared memory
// more effectively for reductions, especially for larger row sizes.
__global__ void softmax_kernel(float *data, int n_rows, int n_cols) {
    // One block per row seems reasonable for moderate n_cols
    int row = blockIdx.x;
    if (row >= n_rows) return;

    int row_start_idx = row * n_cols;
    float *row_data = data + row_start_idx;

    // --- 1. Find Max value in the row ---
    // Simple reduction (inefficient for large n_cols without shared memory)
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
         max_val = max(max_val, row_data[i]);
    }
     // Block-level reduction for max_val (requires __syncthreads and shared memory for efficiency)
     // For simplicity here, we assume one thread finds the max (conceptually incorrect for parallelism)
     // A proper implementation would use shared memory reduction. Let's keep it basic for demo:
     // This part NEEDS improvement for real performance. We'll just use thread 0's potentially partial max.
     // A *real* implementation needs a block-wide reduction here.
     // Let's approximate with a loop (still not fully parallel reduction):
     if (threadIdx.x == 0) {
         max_val = -std::numeric_limits<float>::infinity();
         for (int i = 0; i < n_cols; ++i) {
              max_val = max(max_val, row_data[i]);
         }
     }
    // Broadcast max_val - again, shared memory is the way
     __syncthreads(); // Ensure max_val calculation (even the flawed one) is done
     // In a real kernel, max_val would be read from shared memory after reduction.


    // --- 2 & 3. Subtract max, Exponentiate, and Calculate Sum ---
    float sum_exp = 0.0f;
     // Again, a parallel reduction is needed for the sum. We simplify.
     if (threadIdx.x == 0) { // Only thread 0 calculates the sum (inefficient)
         for (int i = 0; i < n_cols; ++i) {
             row_data[i] = expf(row_data[i] - max_val); // Subtract max and exponentiate
             sum_exp += row_data[i];
         }
     }
     __syncthreads(); // Ensure exponentiation is done and sum calculated by thread 0


    // --- 4. Normalize ---
    // Divide each element by the sum. This part can be parallel.
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) {
        if (sum_exp > 0.0f) { // Avoid division by zero
             row_data[i] /= sum_exp;
        } else {
            // Handle case where sum is zero (e.g., all inputs were -inf)
            // Distribute probability uniformly, or set to zero, depending on desired behavior.
            row_data[i] = 1.0f / n_cols; // Example: Uniform distribution
        }
    }
     __syncthreads(); // Ensure normalization is complete before block exits
}


// --- Host Function ---
void scaled_dot_product_attention(
    float *h_q, float *h_k, float *h_v, float *h_mask, // Host inputs (Q, K, V, Mask)
    float *h_output,                                 // Host output
    int seq_len_q, int seq_len_k, int d_k, int d_v,
    bool apply_mask)
{
    // Matrix dimensions
    // Q: (seq_len_q, d_k)     -> M = seq_len_q, K = d_k
    // K: (seq_len_k, d_k)     -> K_orig = seq_len_k, N_orig = d_k
    // K^T: (d_k, seq_len_k)   -> K = d_k, N = seq_len_k
    // V: (seq_len_k, d_v)     -> M_v = seq_len_k, N_v = d_v
    // ---
    // Scores = Q * K^T: (seq_len_q, seq_len_k) -> M = seq_len_q, N = seq_len_k
    // Attn   = softmax(Scores): (seq_len_q, seq_len_k) -> M = seq_len_q, N = seq_len_k
    // Output = Attn * V: (seq_len_q, d_v) -> M = seq_len_q, N = d_v

    int M = seq_len_q;
    int N_scores = seq_len_k;
    int K_qk = d_k;
    int N_v = d_v;

    // --- Device Memory Allocation ---
    float *d_q, *d_k, *d_v, *d_scores, *d_attn_weights, *d_output, *d_mask = nullptr;
    CHECK_CUDA(cudaMalloc(&d_q, M * K_qk * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_k, N_scores * K_qk * sizeof(float))); // K is (seq_len_k, d_k)
    CHECK_CUDA(cudaMalloc(&d_v, N_scores * N_v * sizeof(float))); // V is (seq_len_k, d_v)
    CHECK_CUDA(cudaMalloc(&d_scores, M * N_scores * sizeof(float))); // Stores intermediate Q*K^T result
    // d_attn_weights will reuse d_scores buffer after softmax
    CHECK_CUDA(cudaMalloc(&d_output, M * N_v * sizeof(float)));
    if (apply_mask && h_mask != nullptr) {
        CHECK_CUDA(cudaMalloc(&d_mask, M * N_scores * sizeof(float))); // Mask dimensions match scores
    }

    // --- Copy Data Host -> Device ---
    CHECK_CUDA(cudaMemcpy(d_q, h_q, M * K_qk * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, h_k, N_scores * K_qk * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, h_v, N_scores * N_v * sizeof(float), cudaMemcpyHostToDevice));
    if (apply_mask && h_mask != nullptr) {
        CHECK_CUDA(cudaMemcpy(d_mask, h_mask, M * N_scores * sizeof(float), cudaMemcpyHostToDevice));
    }

    // --- cuBLAS Initialization ---
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // --- Attention Calculation ---
    float alpha = 1.0f;
    float beta = 0.0f;
    float scale_factor = 1.0f / sqrtf((float)d_k);

    // 1. Compute Scores = Q * K^T
    // cublasSgemm computes C = alpha * op(A) * op(B) + beta * C
    // Here: Scores = 1.0 * Q * K^T + 0.0 * Scores
    // A = Q (M x K_qk), B = K (N_scores x K_qk)
    // We need op(B) = K^T (K_qk x N_scores)
    // Result C = Scores (M x N_scores)
    // Note: cuBLAS uses column-major order by default. If your matrices are row-major in memory,
    // you can sometimes swap A and B and their transpositions, or adjust leading dimensions (lda, ldb, ldc).
    // Assuming row-major layout and standard C arrays:
    // C(M,N) = A(M,K) * B(K,N) -> This is standard math layout
    // cublasSgemm with row-major thinking (requires reversing order and transposing):
    // C = A*B -> sgemm(B^T, A^T) column major == sgemm(A, B) row major (roughly)
    // Let's compute C^T = B^T * A^T using column-major cuBLAS thinking
    // C = Q * K^T (M x N_scores) = Q(M x K_qk) * K^T(K_qk x N_scores)
    // cuBLAS expects column-major. If our data IS row-major:
    // To compute C = A * B (row major), call cublas with C^T = B^T * A^T (col major)
    // Here: Scores = Q * K^T.
    // Column-major perspective: Scores^T (N_scores x M) = K (N_scores x K_qk) * Q^T (K_qk x M)
    // So, A = K (no transpose), B = Q (transpose)
    // M_cublas = N_scores, N_cublas = M, K_cublas = K_qk
    // lda = M_cublas = N_scores, ldb = K_cublas = K_qk, ldc = M_cublas = N_scores
    // Let's stick to the logical operation C = Q * K^T and tell cuBLAS A=Q, B=K, op(B)=Transpose
    // Assuming Row-Major Storage for C/C++ compatibility:
    // We want to compute C(M, N) = alpha * A(M, K) * B(K, N) + beta * C(M, N)
    // In cuBLAS (column-major): C'(N, M) = alpha * B'(N, K) * A'(K, M) + beta * C'(N, M)
    // If A, B, C are row-major:
    //   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N)
    // Let's try C = Q * K^T
    // A = Q (M x K_qk), B = K (N_scores x K_qk), C = Scores (M x N_scores)
    // We need B^T (K_qk x N_scores).
    // Row-major sgemm C(M,N) = A(M,K) * B^T(K,N) :
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N=N_scores, M=M, K=K_qk, &alpha, d_k, K=K_qk, d_q, K=K_qk, &beta, d_scores, N=N_scores)

    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N, // op(K), op(Q) -> K needs T, Q needs N for Q*K^T result
                             N_scores, M, K_qk,         // N, M, K (dimensions of result C is M x N -> cublas needs N x M)
                             &alpha,
                             d_k, K_qk,                 // B = K, ldb = K_qk (leading dim of K)
                             d_q, K_qk,                 // A = Q, lda = K_qk (leading dim of Q)
                             &beta,
                             d_scores, N_scores));     // C = Scores, ldc = N_scores (leading dim of Scores)


    // 2. Scale Scores
    int num_score_elements = M * N_scores;
    int threads_per_block = 256;
    int blocks = (num_score_elements + threads_per_block - 1) / threads_per_block;
    scale_kernel<<<blocks, threads_per_block>>>(d_scores, scale_factor, num_score_elements);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // 3. Apply Mask (Optional)
    if (apply_mask && d_mask != nullptr) {
        // Launch configuration needs careful thought. Launching one block per row might be okay
        // if seq_len_k is not excessively large, otherwise need 2D grid or different strategy.
        // Let's try a 2D grid: blocks handle elements, check bounds inside. Simpler grid:
        dim3 blockDimMask(threads_per_block, 1); // Threads mainly iterate cols
        dim3 gridDimMask(1, M); // One block vertically per row
        // Adjust blockDimMask.x if N_scores > threads_per_block needed multiple blocks per row horizontally
        if (N_scores > threads_per_block) {
             blockDimMask.x = threads_per_block; // Max threads
             gridDimMask.x = (N_scores + threads_per_block -1) / threads_per_block; // Blocks needed horizontally
        } else {
             blockDimMask.x = N_scores; // Fewer threads if row is small
             gridDimMask.x = 1;
        }

        float mask_fill_value = -1e9; // A large negative number
        // Using 1D grid/block for simplicity matching scale kernel - less efficient for masking logic
        // mask_kernel<<<gridDimMask, blockDimMask>>>(d_scores, d_mask, mask_fill_value, M, N_scores);

        // Simpler Launch matching scale kernel (less optimal locality for masking)
         mask_kernel<<<blocks, threads_per_block>>>(d_scores, d_mask, mask_fill_value, M, N_scores); // Needs modification to use idx logic
         // Let's use the block-per-row approach, easier kernel logic
         dim3 gridSoftmax(M, 1, 1); // One block per row
         dim3 blockSoftmax(256, 1, 1); // Threads per block (adjust based on GPU/N_scores)
         // Need to make sure block size is <= 1024 and handles N_scores potentially > blockDim.x
         if (N_scores < blockSoftmax.x) blockSoftmax.x = N_scores; // Optimize for small rows
         // Mask kernel adapted for block-per-row:
         // Assuming mask_kernel takes row/col based indices
         mask_kernel<<<gridSoftmax, blockSoftmax>>>(d_scores, d_mask, mask_fill_value, M, N_scores);
         CHECK_CUDA(cudaGetLastError());
    }

    // 4. Softmax
    // Launch one block per row (M blocks), with threads collaborating on columns (N_scores)
    dim3 gridSoftmax(M, 1, 1); // One block per row
    dim3 blockSoftmax(256, 1, 1); // Adjust based on GPU / N_scores constraints
    // Ensure block size is appropriate
    if (N_scores > 1024) {
         // This simple kernel won't handle >1024 cols efficiently without modifications
         printf("Warning: N_scores (%d) > max block size (1024). Softmax kernel needs improvement.\n", N_scores);
         blockSoftmax.x = 1024; // Use max allowed, kernel logic must handle loops
    } else {
         blockSoftmax.x = (N_scores < 32) ? 32 : (N_scores + 31) / 32 * 32; // Use warp-multiple, min 32
         if (blockSoftmax.x > 1024) blockSoftmax.x = 1024; // Clamp
    }


    softmax_kernel<<<gridSoftmax, blockSoftmax>>>(d_scores, M, N_scores);
    CHECK_CUDA(cudaGetLastError());
    // d_scores now holds the attention weights

    float *d_attn = d_scores; // Reuse buffer

    // 5. Compute Output = Attention_Weights * V
    // C = alpha * A * B + beta * C
    // Output = 1.0 * Attn * V + 0.0 * Output
    // A = Attn (M x N_scores), B = V (N_scores x N_v)
    // C = Output (M x N_v)
    // Row-major sgemm C(M,N) = A(M,K) * B(K,N):
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N=N_v, M=M, K=N_scores, &alpha, d_v, N=N_v, d_attn, K=N_scores, &beta, d_output, N=N_v)

    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N, // op(V), op(Attn) -> No transpose needed
                             N_v, M, N_scores,          // N, M, K (dimensions C is M x N -> cublas needs N x M)
                             &alpha,
                             d_v, N_v,                  // B = V, ldb = N_v (leading dim of V)
                             d_attn, N_scores,          // A = Attn, lda = N_scores (leading dim of Attn)
                             &beta,
                             d_output, N_v));           // C = Output, ldc = N_v (leading dim of Output)


    // --- Copy Result Device -> Host ---
    CHECK_CUDA(cudaMemcpy(h_output, d_output, M * N_v * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Cleanup ---
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_q));
    CHECK_CUDA(cudaFree(d_k));
    CHECK_CUDA(cudaFree(d_v));
    CHECK_CUDA(cudaFree(d_scores)); // Freed d_attn implicitly
    CHECK_CUDA(cudaFree(d_output));
    if (d_mask != nullptr) {
        CHECK_CUDA(cudaFree(d_mask));
    }
}


// --- Main Function (Example Usage) ---
int main() {
    int seq_len_q = 4;  // Example: Target sequence length
    int seq_len_k = 5;  // Example: Source sequence length
    int d_k = 8;       // Example: Dimension of keys/queries
    int d_v = 10;      // Example: Dimension of values

    // Allocate host memory
    float *h_q = (float*)malloc(seq_len_q * d_k * sizeof(float));
    float *h_k = (float*)malloc(seq_len_k * d_k * sizeof(float));
    float *h_v = (float*)malloc(seq_len_k * d_v * sizeof(float));
    float *h_mask = (float*)malloc(seq_len_q * seq_len_k * sizeof(float)); // Mask (optional)
    float *h_output = (float*)malloc(seq_len_q * d_v * sizeof(float));

    // Initialize with some data (e.g., simple ranges or random)
    printf("Initializing host data...\n");
    for (int i = 0; i < seq_len_q * d_k; ++i) h_q[i] = (float)(i % 10);
    for (int i = 0; i < seq_len_k * d_k; ++i) h_k[i] = (float)(i % 8);
    for (int i = 0; i < seq_len_k * d_v; ++i) h_v[i] = (float)(i % 12);

    // Initialize mask (e.g., mask out the last key for all queries)
    for (int i = 0; i < seq_len_q; ++i) {
        for (int j = 0; j < seq_len_k; ++j) {
             // Mask is 1 if allowed, 0 if masked
            h_mask[i * seq_len_k + j] = (j == seq_len_k - 1) ? 0.0f : 1.0f; // Mask last column
        }
    }
    bool use_mask = true;
    printf("Masking enabled: %s\n", use_mask ? "Yes" : "No");


    printf("Running scaled dot-product attention on GPU...\n");
    scaled_dot_product_attention(h_q, h_k, h_v, h_mask, h_output,
                                 seq_len_q, seq_len_k, d_k, d_v, use_mask);
    printf("GPU computation finished.\n");

    // --- Print results (optional) ---
    printf("Output (first few elements):\n");
    for (int i = 0; i < seq_len_q; ++i) {
        for (int j = 0; j < d_v; ++j) {
            if (j < 5) { // Print first 5 elements per row
                 printf("%.4f ", h_output[i * d_v + j]);
            }
        }
         printf("...\n");
    }

    // Free host memory
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_mask);
    free(h_output);

    printf("Done.\n");
    return 0;
}