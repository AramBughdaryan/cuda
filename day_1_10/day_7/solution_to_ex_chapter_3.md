Exercise 3.
If the SM of a CUDA device can take up to 1536 threads and up to 4 thread
blocks. Which of the following block configuration would result in the largest
number of threads in the SM?
A. 128 threads per block
B. 256 threads per block
C. 512 threads per block
D. 1024 threads per block

Solution.
Since we can have at most 1536 threads and up to 4 blocks. ideally we would like to use 1536/4 threads per block.
if we use 128 then we need 1536 / 128 = 12 thread blocks which we can not do in parallel.
if we use 256 then we need 1536 / 256 = 6 thread blocks which we can not do in parallel.
if we use 512 then we need [1536 / 512] = 3 thread blocks which **can** do in parallel.
in case of 1024 we would have just one block

Exercise 4.
For a vector addition, assume that the vector length is 2000, each thread
calculates one output element, and the thread block size is 512 threads. How
many threads will be in the grid?
A. 2000
B. 2024
D. 2048
D. 2096

Solution.
512 * 4 = 2048

Exercise 5.
With reference to the previous question, how many warps do you expect to
have divergence due to the boundary check on vector length?

Solution.
We can see the last warp is totlay out of calculation so it has nothing to compute => it has no divergence.
but pre-last one has divergence since some of it is doing calculation and other part is not doing. => answer is 1.

Exercise 6.
You need to write a kernel that operates on an image of size 400 × 900 pixels.
You would like to assign one thread to each pixel. You would like your thread
blocks to be square and to use the maximum number of threads per block
possible on the device (your device has compute capability 3.0). How would
you select the grid dimensions and block dimensions of your kernel?

Solution.
I will select blocks of size (32, 32) and will take grid of size (13, 29).

Exercise 7.
With reference to the previous question, how many idle threads do you expect
to have?

Solution.
I will have total of 13 * 29 * 32 * 32 = 386048 threads. From which only 400 * 900 are doing actual computation.
386048 - 400 * 900 = 26048 threads are going to be idle.


Exercise 7.
Consider a hypothetical block with 8 threads executing a section of code
before reaching a barrier. The threads require the following amount of time
(in microseconds) to execute the sections: 2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, and
2.9 and to spend the rest of their time waiting for the barrier. What percentage
of the total execution time of the thread is spent waiting for the barrier?

Solution.
Total execution time is 3 μs since every thread will wait to the one for which it takes the longest time.
We need to calculate average time of waiting.
Let’s compute them all:

Thread 0: 3.0−2.0=1.0 μs
Thread 1: 3.0−2.3=0.7 μs
Thread 2: 3.0−3.0=0.0 μs
Thread 3: 3.0−2.8=0.2 μs
Thread 4: 3.0−2.4=0.6 μs
Thread 5: 3.0−1.9=1.1 μs
Thread 6: 3.0−2.6=0.4 μs
Thread 7: 3.0−2.9=0.1 μs

Which in average is 4.1 / 8 = 0.5125 μs
and **0.5125** is around 17.1% of 3.

Exercise 9
Indicate which of the following assignments per multiprocessor is possible.
In the case where it is not possible, indicate the limiting factor(s).
A. 8 blocks with 128 threads each on a device with compute capability 1.0
B. 8 blocks with 128 threads each on a device with compute capability 1.2
C. 8 blocks with 128 threads each on a device with compute capability 3.0
D. 16 blocks with 64 threads each on a device with compute capability 1.0
E. 16 blocks with 64 threads each on a device with compute capability 1.2
F. 16 blocks with 64 threads each on a device with compute capability 3.0

Solution.
Compute Capability 1.0:
Maximum resident threads per SM: 768
Maximum resident blocks per SM: 8
Compute Capability 1.2:
Maximum resident threads per SM: 1024
Maximum resident blocks per SM: 8
Compute Capability 3.0:
Maximum resident threads per SM: 2048
Maximum resident blocks per SM: 16

So every point exect of F which want to use 16 blocks is not possible.
and based on number of threads we can see that in A 8 * 128 > 768 => it's not possible.

Excercise 10.
A CUDA programmer says that if they launch a kernel with only 32 threads
in each block, they can leave out the __syncthreads() instruction wherever
barrier synchronization is needed. Do you think this is a good idea? Explain.

Solutions.
Even though a block of 32 threads maps to one warp and might appear to execute synchronously, relying on this fact instead of using __syncthreads() is risky. It compromises portability, correctness in the presence of divergence, memory consistency, and overall code clarity. Therefore, it's best to include __syncthreads() wherever barrier synchronization is needed.

Exercise 11.
A student mentioned that he was able to multiply two 1024 × 1024 matrices
by using a tiled matrix multiplication code with 32 × 32 thread blocks.
He is using a CUDA device that allows up to 512 threads per block and up
to 8 blocks per SM. He further mentioned that each thread in a thread block
calculates one element of the result matrix. What would be your reaction
and why?

Solution.
The student's configuration is problematic. Using a 32 × 32 thread block means that each block contains 32 * 32 = 1024 threads.
but his CUDA device allows up to 512! So he needs to use 16 * 16 block size or any other (i, j) such that i * j <= 512.
