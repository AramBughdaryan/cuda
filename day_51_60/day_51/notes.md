This chapter section describes a method proposed by Siebert and Traff (2012) for parallelizing the merge operation of two sorted arrays, A (size m) and B (size n), into a single sorted output array, C (size m+n).

The core idea is to divide the work among multiple threads by assigning each thread a specific range of indices in the output array C that it is responsible for generating.

To achieve this in parallel without conflicts, a key concept called the "co-rank" function is introduced. For any element C[k] at index k (its rank) in the final merged array, there are unique indices i in A and j in B such that k = i + j, and the prefix C[0...k-1] is the result of merging the prefixes A[0...i-1] and B[0...j-1]. These unique indices i and j are the co-ranks corresponding to rank k.

The parallelization process works as follows:

The output array C is conceptually divided into contiguous segments, one for each thread.
Each thread knows the range of ranks (output indices) it needs to produce.
Using the co-rank function, each thread determines the corresponding ranges of indices within the input arrays A and B that contain the necessary elements to produce its assigned output segment.
Since each thread now knows its specific input subarrays from A and B and its specific output subarray in C, it can perform a standard sequential merge operation on these subarrays independently and in parallel with other threads.
The chapter highlights that this approach is distinct from simpler parallelization patterns because the input data range for each thread cannot be determined by simple index calculations alone; it depends on the actual values within the input arrays, making the co-rank function essential and the parallel merge pattern more complex.