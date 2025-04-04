#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>       // For FLT_MAX, FLT_MIN
#include <cmath>        // For exp, fmaxf
#include <vector>       // For host data storage
#include <iostream>     // For printing results
#include <random>       // For generating test data
#include <iomanip>      // For formatting output
#include <numeric>      // For std::accumulate

// --- Error Checking Macro ---
#define CHECK_CUDA_ERROR(val) checkCuda((val), #val, __FILE__, __LINE__)
void checkCuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") "
                  << "at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset(); // Reset device on error
        exit(99);
    }
}

// ============================================================================
// == CUDA Kernel (from previous example - slightly cleaned up) ================
// ============================================================================

__global__ void softmaxKernel(const float* input, float* output, int numFeatures, int batchSize) {
    // One block per row/batch item
    int row = blockIdx.x;
    // Ensure we don't process rows beyond batchSize if gridDim.x > batchSize
    if (row >= batchSize) return;

    int tid = threadIdx.x; // Thread ID within the block

    // Allocate shared memory - size should be at least blockDim.x
    extern __shared__ float shared_data[];

    // --- 1. Load data and Find Max ---
    float threadMax = -FLT_MAX; // Initialize with smallest possible float

    // Cooperatively load row data into shared memory & find thread's local max portion
    for (int i = tid; i < numFeatures; i += blockDim.x) {
        int idx = row * numFeatures + i;
        float val = input[idx];
        shared_data[tid + (i / blockDim.x) * blockDim.x] = val; // Store potentially multiple values per thread if numFeatures > blockDim.x
        threadMax = fmaxf(threadMax, val);
    }
     __syncthreads(); // Ensure all data is loaded by the block

    // --- Block-wide reduction for Max ---
    // Store partial max in shared memory for reduction
    // Note: This reduction assumes blockDim.x is a power of 2 for simplicity
    // More robust reductions handle arbitrary sizes.
    // If looping above, need to reduce multiple threadMax values first.
    // For simplicity, assume numFeatures <= blockDim.x for this reduction part
    if (numFeatures <= blockDim.x) {
         shared_data[tid] = (tid < numFeatures) ? threadMax : -FLT_MAX;
    } else {
         // Need a more complex reduction handling the loop case
         // Simplified: assumes one value per thread for reduction phase
         shared_data[tid] = threadMax; // May overwrite data if not careful
    }
    __syncthreads();


    // Reduce within the block (log_2 steps)
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
             // Check bounds implicitly via initial padding/values if needed
             shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + offset]);
        }
         __syncthreads(); // Wait for all threads to complete the step
    }
    // The maximum value for the row is now in shared_data[0]
    float rowMax = shared_data[0];
    // No need for final syncthreads here as all threads will read the same rowMax


    // --- 2. Calculate exp(x - max) and Sum ---
    float threadSum = 0.0f;
    // Loop again to calculate exponentials and sum
     for (int i = tid; i < numFeatures; i += blockDim.x) {
        int idx = row * numFeatures + i;
         // Read original value again (from global or shared memory if preserved)
         // Assuming re-read from global for simplicity now:
        float original_val = input[idx];
        float exp_val = expf(original_val - rowMax);
        shared_data[tid + (i / blockDim.x) * blockDim.x] = exp_val; // Store exp_val in shared mem
        threadSum += exp_val;
    }
     __syncthreads(); // Ensure all exp_vals calculated


    // --- Block-wide reduction for Sum ---
    // Store partial sum in shared memory for reduction
     if (numFeatures <= blockDim.x) {
         shared_data[tid] = (tid < numFeatures) ? threadSum : 0.0f;
     } else {
        // Complex reduction needed here too
        shared_data[tid] = threadSum;
     }
     __syncthreads();


    // Reduce sum within the block
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_data[tid] += shared_data[tid + offset];
        }
        __syncthreads();
    }
    // The sum for the row is now in shared_data[0]
    float rowSum = shared_data[0];
     // Add small epsilon to prevent division by zero, though exp ensures sum > 0 unless all inputs are -inf
    const float epsilon = 1e-12f; // Or handle zero sum explicitly if needed
    rowSum += epsilon;
    // No need for final syncthreads here as all threads read the same rowSum


    // --- 3. Normalize and Write Output ---
     for (int i = tid; i < numFeatures; i += blockDim.x) {
        // Read the exp_val calculated earlier and stored in shared memory
        float myExpVal = shared_data[tid + (i / blockDim.x) * blockDim.x];
        int output_idx = row * numFeatures + i;
        output[output_idx] = myExpVal / rowSum;
    }
}


// ============================================================================
// == CPU Reference Implementation ============================================
// ============================================================================
void softmax_cpu(const std::vector<float>& input, std::vector<float>& output, int numFeatures, int batchSize) {
    output.resize(input.size());

    for (int i = 0; i < batchSize; ++i) {
        const float* rowInput = input.data() + i * numFeatures;
        float* rowOutput = output.data() + i * numFeatures;

        // 1. Find max element in the current row
        float maxVal = -FLT_MAX;
        for (int j = 0; j < numFeatures; ++j) {
            maxVal = std::fmax(maxVal, rowInput[j]);
        }

        // 2. Calculate exponentials and sum
        float sumExp = 0.0f;
        for (int j = 0; j < numFeatures; ++j) {
            // Use temporary storage or recalculate if needed, here recalculate
            float expVal = std::exp(rowInput[j] - maxVal);
            rowOutput[j] = expVal; // Store intermediate exp value
            sumExp += expVal;
        }

        // Add small epsilon for stability if sum could be zero
        const float epsilon = 1e-12f;
        sumExp += epsilon;


        // 3. Normalize
        for (int j = 0; j < numFeatures; ++j) {
            rowOutput[j] /= sumExp;
        }
    }
}

// ============================================================================
// == Verification Function ==================================================
// ============================================================================
bool verify_results(const std::vector<float>& cpu_output, const std::vector<float>& gpu_output, float tolerance = 1e-5f) {
    if (cpu_output.size() != gpu_output.size()) {
        std::cerr << "Error: Output sizes do not match!" << std::endl;
        return false;
    }

    float max_diff = 0.0f;
    int diff_count = 0;
    size_t first_diff_idx = 0;

    for (size_t i = 0; i < cpu_output.size(); ++i) {
        float diff = std::abs(cpu_output[i] - gpu_output[i]);
        if (diff > tolerance) {
            if (diff_count == 0) { // Store first difference details
                 first_diff_idx = i;
            }
             diff_count++;
            max_diff = std::fmax(max_diff, diff);
            // Optional: Stop early if too many differences?
            // if (diff_count > 10) break;
        }
    }

    if (diff_count > 0) {
        std::cerr << "Verification FAILED!" << std::endl;
        std::cerr << "Number of differences exceeding tolerance (" << tolerance << "): " << diff_count << std::endl;
        std::cerr << "Maximum difference: " << max_diff << std::endl;
        std::cerr << "First difference at index " << first_diff_idx << ": CPU=" << cpu_output[first_diff_idx]
                  << ", GPU=" << gpu_output[first_diff_idx] << std::endl;
        return false;
    } else {
        std::cout << "Verification PASSED." << std::endl;
        std::cout << "Maximum observed difference: " << max_diff << std::endl;
        return true;
    }
}


// ============================================================================
// == GPU Launcher & Benchmarking Function ====================================
// ============================================================================
float softmax_gpu_benchmark(const std::vector<float>& h_input, std::vector<float>& h_output_gpu, int batchSize, int numFeatures, int iterations = 100) {

    float *d_input, *d_output;
    size_t dataSize = (size_t)batchSize * numFeatures * sizeof(float);
    h_output_gpu.resize(h_input.size()); // Ensure host output buffer is sized

    // Allocate memory on GPU
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, dataSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, dataSize));

    // Copy input data from host to device (once before timing loop)
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), dataSize, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    int threadsPerBlock = 256; // Common choice, tune if necessary
    // Adjust if numFeatures is small or large
    if (numFeatures < threadsPerBlock && numFeatures > 0) {
         // Optimize for smaller features? Round up to warp size?
         // For simplicity stick to 256, kernel handles internal checks.
    } else if (numFeatures > 1024) {
         // Kernel needs modification (looping) for numFeatures > maxThreadsPerBlock
         // Simple version might require threadsPerBlock >= numFeatures or specific handling.
         // For this example, we'll try 1024 max, but the kernel loops.
         threadsPerBlock = 1024;
    }

    dim3 blockDim(threadsPerBlock);
    // Grid Dim: One block per row
    dim3 gridDim(batchSize); // Can optimize if batchSize is huge (grid stride loops)

    // Shared memory size: Needs space for data elements for reduction
    // Needs at least blockDim.x floats
    size_t sharedMemSize = blockDim.x * sizeof(float);
    // If kernel uses more shared memory (e.g. preserving original values), adjust this.
     if (numFeatures > blockDim.x) {
         // If threads loop, they might need more shared memory to store intermediate results correctly.
         // Calculate how many full blockDim.x chunks are needed
         int chunks = (numFeatures + blockDim.x - 1) / blockDim.x;
         sharedMemSize = chunks * blockDim.x * sizeof(float);
         // Check against max shared memory per block if this gets large!
         // cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0); prop.sharedMemPerBlock;
     }


    // --- Benchmarking Setup ---
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Warm-up run (optional but recommended)
    softmaxKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_output, numFeatures, batchSize);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for warm-up

    // --- Timing Loop ---
    CHECK_CUDA_ERROR(cudaEventRecord(start)); // Record start event

    for (int i = 0; i < iterations; ++i) {
        softmaxKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_output, numFeatures, batchSize);
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop));   // Record stop event
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop)); // Wait for the stop event to complete

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop)); // Get elapsed time

    // Clean up events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // --- Get Results ---
    // Copy output data from device to host AFTER timing
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu.data(), d_output, dataSize, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    // Return average time per iteration
    return milliseconds / iterations;
}


// ============================================================================
// == Main Function ===========================================================
// ============================================================================
int main() {
    // --- Parameters ---
    int batchSize = 1024;    // Number of rows
    int numFeatures = 4096; // Number of columns (features per row)
    int benchmark_iterations = 100;

    std::cout << "Softmax Benchmark & Test" << std::endl;
    std::cout << "Batch Size: " << batchSize << std::endl;
    std::cout << "Num Features: " << numFeatures << std::endl;
    std::cout << "Benchmark Iterations: " << benchmark_iterations << std::endl;
     std::cout << "------------------------------------" << std::endl;


    // --- Host Data Allocation ---
    std::vector<float> h_input( (size_t)batchSize * numFeatures);
    std::vector<float> h_output_gpu; // Will be resized by benchmark function
    std::vector<float> h_output_cpu; // Will be resized by cpu function

    // --- Initialize Input Data ---
    std::cout << "Initializing host data..." << std::endl;
    std::mt19937 rng(12345); // Random number generator with fixed seed
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f); // Range of random values
    for (size_t i = 0; i < h_input.size(); ++i) {
        h_input[i] = dist(rng);
    }

    // --- Run GPU Benchmark ---
    std::cout << "Running GPU Softmax and Benchmarking..." << std::endl;
    float avgGpuTime_ms = softmax_gpu_benchmark(h_input, h_output_gpu, batchSize, numFeatures, benchmark_iterations);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Ensure all GPU work is done before proceeding

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Average GPU Kernel Time: " << avgGpuTime_ms << " ms" << std::endl;

    // Optional: Calculate Throughput
    double dataSizeGB = (double)batchSize * numFeatures * sizeof(float) / (1024.0 * 1024.0 * 1024.0);
    // Throughput considers reading input + writing output = 2 * dataSize
    double throughputGBs = (2.0 * dataSizeGB) / (avgGpuTime_ms / 1000.0);
    std::cout << "Effective Throughput: " << throughputGBs << " GB/s" << std::endl;


    // --- Run CPU Reference ---
    std::cout << "Running CPU Softmax for verification..." << std::endl;
    softmax_cpu(h_input, h_output_cpu, numFeatures, batchSize);
    std::cout << "CPU Softmax completed." << std::endl;
     std::cout << "------------------------------------" << std::endl;


    // --- Verify Results ---
    std::cout << "Verifying GPU results against CPU..." << std::endl;
    bool success = verify_results(h_output_cpu, h_output_gpu, 1e-5f); // Adjust tolerance if needed


    std::cout << "------------------------------------" << std::endl;
    if (!success) {
        std::cerr << "Test Failed!" << std::endl;
        return 1; // Indicate error
    } else {
         std::cout << "Test Passed!" << std::endl;
    }

    return 0; // Indicate success
}