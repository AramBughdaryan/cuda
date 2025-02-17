**Summary of Convolution:**

Convolution is an array operation where each output element is computed as a weighted sum of neighboring input elements, using a **convolution mask** (or kernel) to define the weights. Key points include:

1. **1D Convolution**:  
   - Used in audio processing (1D signals).  
   - Example: A 5-element mask computes each output element by centering the mask on the corresponding input element, extending symmetrically (e.g., two neighbors on each side).  
   - **Boundary Handling**: For edge elements (e.g., P[1]), missing neighbors (ghost cells) are typically assigned default values (often 0).  

2. **2D Convolution**:  
   - Applied in image processing (2D grids).  
   - A 2D mask (e.g., 5x5) slides over the input array, computing output pixels via weighted sums of subarrays.  
   - **Boundaries**: Missing rows/columns at edges (e.g., P₁,₀) are handled similarly, with ghost cells often set to 0.  

3. **General Principles**:  
   - **Odd-sized masks** ensure symmetry around the target element.  
   - **Ghost cells** impact parallel computation efficiency, especially in tiling strategies.  
   - Applications may vary in handling ghost cells (e.g., using zeros or replicating edge values).  

4. **Example Calculations**:  
   - In 1D, P[2] = 1×3 + 2×4 + 3×5 + 4×4 + 5×3 = 57.  
   - In 2D, P₂,₂ involves summing products of a 5x5 input subarray and mask.  

**Note**: The term "convolution mask" avoids confusion with CUDA kernels (parallel functions). Efficient implementation must address boundary conditions and ghost cells, which influence memory access patterns in parallel architectures.