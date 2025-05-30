# 100 Days of GPU Challenge  
*(Huge thanks to [Umar Jamil](https://github.com/hkproj/100-days-of-gpu) for organizing this challenge!)*

## Progress Log

### **Day 1**
- Learned the basic syntax of C++ and CUDA.
- Implemented a simple kernel for increment.
  *(Later, I added scripts to call this kernel from PyTorch.)*
- Completed **Chapter 1**.

### **Day 2**
- Implemented both GPU and CPU versions of vector addition.
  *(Unfortunately, the GPU version was slower for this simple task.)*
- Completed **Chapter 2**.

### **Day 3**
- Implemented `colorToGreyscaleConversion`.
- Started reading **Chapter 3** (currently on page 52, with some questions).

### **Day 4**  
- Resolved questions from **Day 3**.
- Implemented an **image blurring kernel**.
- Dived into device properties section.
- Almost finished with **Chapter 3** (currently on page 64!).
- Wrote simple example how to use `image_blur.cu`.
- created helpers folder and some functions for future use.

### **Day 5**
- Finish the chapter 3
- Matrix Addition kernel

### **Day 6**
- Matrix addition kernel by row and column
- matrixMulVectorKernel 
- Exercises 1-6 solved! (Page 68)

### **Day 7**
- Exercises 6-11 solved!(page 69)
- Implement simple matrix multiplication kernel
- Start Chapter 4. Until section 4.4

### **Day 8**
- Read section 4.4
- add test for simple kernel from **day 7**
- start implementation of **tiledMatrixMulKernel**

### **Day 9**
- Implemented and finally understood **tiledMatrixMulKernel**
- Watched Lecture 5 on PMPP book from **https://www.youtube.com/watch?v=31ZyYkoClT4&list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4&index=5**
- Some test to validate results.
- Read section 4.5

### **Day 10**
- Fixed **tiledMatrixMulKernel** to work with boundary conditions
- Read section 4.6

### **Day 11**
- Finished Reading chapter 4 of PMPP book.
- Exercises 1-9

### **Day 12**
- Exercise 10 of chapter 4.

### **Day 13**
- Read section 5.1 on Global Memoery Bandwidth
- Learn about memory coalesced
- Come back with Triton. vector addition + benchmarking

### **Day 14**
- Read section 5.2
- Fused softmax with Triton + Benchmarking with naive softmax and PyTorch's softmax

### **Day 15**
- Read part of section 5.3 till page 122
- Implement simple partialSum kernel
- Implement optimised partialSum (tensor size 1024) (With less thread divergence)
  - Time for partialSumKernel: 0.164544 ms
  - Time for optimisedPartialSumKernel: 0.019264 ms

### **Day 16**
- Finish chapter 5!
- Starting exercises

### **Day 17**
- Finish chapter 6!

### **Day 18**
- Read section 7.1

### **Day 19**
- Read section 7.2
- Implement 1D convolution kernel

### **Day 20**
- Read section 7.3
- Implement 1D convolution kernel with usage of constant memory. (Optimised on speed around 4 times)

### **Day 21**
- Implement `convolution_1D_tiled_kernel`

### **Day 22**
- Read Section 7.4 of PMPP book.

### **Day 23**
- Read Section 7.5 of PMPP.
- Implement with simplified kernel for convolution with hope that L2 cache will be used by gpu

### **Day 24**
- Finish chapter 7 of PMPP.
- Implement 2D convolution.

### **Day 25**
- Read Section 8.1 of PMPP

### **Day 26**
- Read Section 8.2 of PMPP
- Implement Kogge Stone scan kernel

### **Day 27** 
- Start matmul implementation on triton

### **Day 28** 
- Benchmarked implementation of matmul with triton!

### **Day 29**
- Implement Brent-Kung scan kernel.
- Read section 8.4.

### **Day 30**
- Read section 8.5.

### **Day 30**
- Read section 8.6.

### **Day 31**
- Start reading section 8.7

### **Day 32**
- Finish chapter 8

### **Day 33**
- Start solving exercises (1/10)
- Read intro of Chapter 9

### **Day 34**
- Performed analysis of Brent-Kung and Kogge-Stone scan algorithms

### **Day 35**
- Implemented hierarchical parralel scan with some bug (need to resolve)

### **Day 36**
- Read section 9.1

### **Day 37**
- Read section 9.2

### **Day 38**
- Read section 9.3
- Implement histogram calculation with **block** partitioning

### **Day 39**
- Implement histogram calculation with **interleaved** partitioning

### **Day 40**
- Read section 9.4

### **Day 41**
- Read section 9.5-9.6
- Implement histogram computation kernel with privatized memory

### **Day 42**
- Finish chapter 9!
- Implement histogram computation kernel with privatized memory and aggregation for large concentration values

### **Day 43**
- Start implementation of **Linear** layer

### **Day 44**
- Add cublas usage to my Linear layer which still does not work

### **Day 45**
- Fix Linear building
- Implement ReLU activation and benchmarked with PyTorch's ReLU

### **Day 46**
- Read section 12.1, 12.2 (PMPP 4th Edition )
- Implement sequential merge

### **Day 47**
- Implement standard embeeding lookup kernel

### **Day 48**
**Texture Embedding Lookup**
- Implemented texture embedding lookup using `cudaArrays`.  
  *Note: `cudaArrays` are read-only.*

- **Standard Embedding Lookup Kernel vs. Texture Lookup:**  
  Achieved a **1.25x speedup** with the following configuration:
  - **Batch Size:** 1024
  - **Vocabulary Size:** 10,000
  - **Embedding Dimension:** 128

### **Day 49**
- Implemented Embedding layer forward and backward
- Benchmark with pytorch's c++ api

### **Day 50**
- Write wrapper for Embedding layer

### **Day 51**
- Understanding merge sort basics section 12.3
