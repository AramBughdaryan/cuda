Exercise 1.
Consider matrix addition. Can one use shared memory to reduce the global
memory bandwidth consumption? Hint: Analyze the elements accessed by
each thread and see if there is any commonality between threads.

Solution.
No we can not because in case of matrix addition every element is accessed just once.
Moreover any attempt to use shared memory here would incur overhead (extra copy steps) without reducing the number of global accesses.

Exercise 3.
What type of incorrect execution behavior can happen if one or both
__syncthreads() are omitted in the kernel of Fig. 4.16?

Solution.
if first __syncthreads() is omitted then some threads can start calculateing their phase dot product 
before the data is in the acessed locations.
if second __syncthreads() is omitted() then elements of Mds and Nds can be changed before all threads have used them.


Exercise 4.
Assuming that capacity is not an issue for registers or shared memory, give
one important reason why it would be valuable to use shared memory instead
of registers to hold values fetched from global memory? Explain your answer.

Solution.
A key reason is data sharing across multiple threads in a block. Registers in a GPU are private to each thread.

Exercise 5.
For our tiled matrix–matrix multiplication kernel, if we use a 32x32 tile, what
is the reduction of memory bandwidth usage for input matrices M and N?

Solution.
We load 32x32 tile into shared memory and the each element of that 32x32 we use exactly 32 times
for and we don't load that anymore but just read from shared memory. Similar thing happens with N.
Since we read once but use 32 times => 1/32 of the original usage


Exercise 6.
Assume that a CUDA kernel is launched with 1,000 thread blocks, with each
having 512 threads. If a variable is declared as a local variable in the kernel,
how many versions of the variable will be created through the lifetime of the
execution of the kernel?

Soultion.
Since local variable is private for each thread => 512 * 1000 = 512 000 versions.

Exercise 7.
In the previous question, if a variable is declared as a shared memory
variable, how many versions of the variable will be created throughout the
lifetime of the execution of the kernel?

Solution.
Since it's shared variable it will be declared for each block => 1000 versions.

Exercise 8.
Consider performing a matrix multiplication of two input matrices with
dimensions NxN. How many times is each element in the input matrices
requested from global memory in the following situations?
A. There is no tiling.
B. Tiles of size TxT are used.

Solution A.
When there is no tiling we acess to each element of first matrix with number of columns in second matrix, thus N.
Similarly we acess to second matrix's element with number of rows of first matrix, thus N.

Solution B.
When there is a tiling of TxT size.
When we use tiling of size TxT for each element we will acess it [N/T]

A kernel performs 36 floating-point operations and 7 32-bit word global
memory accesses per thread. For each of the following device properties,
indicate whether this kernel is compute- or memory-bound.
A. Peak FLOPS= 200 GFLOPS, Peak Memory Bandwidth= 100 GB/s
B. Peak FLOPS= 300 GFLOPS, Peak Memory Bandwidth= 250 GB/s

Solution A.
Since it processes 7 words of 32 bit (=4 bytes) => 7 * 4 bytes = 28 bytes per thread.

36/28 FLOPs/bytes ~=1.29 FLOPs/byte

It can achive 200 G FLOPs / 100 GB/s = 2 FLOPs/byte

Since the kernel’s arithmetic intensity (1.29) is less than the machine’s ratio (2),
the kernel does fewer computations per byte of data than the hardware is capable of supporting.
This implies that the kernel is not doing enough computation per memory access to keep the compute units busy.

Solution B.

It can achive 300 G FLOPs / 250 GB/s = 1.2 FLOPs/byte

Since 1.29 > 1.2 then it's memory bound.