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

