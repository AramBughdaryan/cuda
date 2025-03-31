# Chapter 8 Summary

In this chapter, we studied scan as an important parallel computing pattern. Scan
enables parallel allocation of resources to parties whose needs are not uniform. The
process converts a seemingly sequential recursive computation into a parallel com-
putation, which helps reduce sequential bottlenecks in various applications. We show
that a simple sequential scan algorithm performs only N additions for an input of N
elements.
We first introduced a parallel Kogge–Stone scan algorithm that is fast and con-
ceptually simple but not work-efficient. As the data set size increases, the number
of execution units needed for a parallel algorithm to break even with the simple
sequential algorithm also increases. For an input of 1024 elements, the parallel algo-
rithm performs over nine times more additions than the sequential algorithm. The
algorithm also requires at least nine times more execution resources to break even
with the sequential algorithm. Thus, Kogge–Stone scan algorithms are typically used
within modest-sized scan blocks.
We then presented a parallel Brent–Kung scan algorithm that is conceptually more
complicated than the Kogge–Stone algorithm. Using a reduction tree phase and a distri-
bution tree phase, the algorithm performs only 2*N− 3 additions regardless of the size
of the input data set. With its number of operations increasing linearly with the size of
the input set, thus work-efficient algorithm is often referred to as data-scalable algorithm.
Unfortunately, due to the nature of threads in a CUDA device, the resource consumption
of a Brent–Kung kernel ends up very similar to that of a Kogge–Stone kernel. A three-
phase scan algorithm that employs corner turning and barrier synchronization proves to
be effective in addressing the work-efficiency problem.
We also presented a hierarchical approach to extending the parallel scan algo-
rithms in order to manage arbitrary-sized input sets. Unfortunately, a straightforward,
three-kernel implementation of the hierarchical scan algorithm incurs redundant
global memory accesses whose latencies are not overlapped with computation. We
show that one can use a stream-based hierarchical scan algorithm to enable a single-
pass, single kernel implementation and improve the global memory access efficiency
of the hierarchical scan algorithm. However, this algorithm requires a carefully
designed adjacent block synchronization using atomic operations, thread memory
fence, and barrier synchronization. In addition, special care is needed to prevent
deadlocks using dynamic block index assignment.