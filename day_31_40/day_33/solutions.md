## Problem 1
In this kernel the scan is implemented by having each thread update its own element based on an element a fixed “stride” behind it. During each iteration of the loop, the code
```
if (threadIdx.x >= stride)
    XY[threadIdx.x] += XY[threadIdx.x - stride];
```
decides which threads perform an addition and which do not.

How Control Divergence Arises
- Divergence in the First Warp:
Let's consider a warp size of 32. For each stride value ￼ in the loop, threads with threadIdx.x < s do not execute the addition (they don’t satisfy the condition), while threads with threadIdx.x \ge s do execute it. In the first warp (thread indices 0–31), if ￼ is one of 1, 2, 4, 8, or 16 (all up to half the warp size), then:
	•	For stride = 1:
	•	Thread 0: ￼ → does not update.
	•	Threads 1–31: ￼ → update.
	•	For stride = 2:
	•	Threads 0–1: ￼ → do not update.
	•	Threads 2–31: update.
	•	The same pattern holds for strides 4, 8, and 16: a subset (from thread 0 up to ￼) does not update, while the remaining threads (up to thread 31) do.
Thus, within the first warp, threads are split into two groups with different execution paths, leading to divergence.

