### **Chapter 12 Summary: Merge Operation**  

#### **12.1 Background**  
- **Ordered Merge Function**: Combines two sorted arrays (**A** and **B**) into a single sorted array (**C**).  
- **Assumptions**:  
  - Input arrays are sorted based on a key and an ordering relation (e.g., **≤** for numerical values).  
  - Stability is preserved: If keys are equal, elements from **A** appear before those from **B** in **C**.  
- **Importance**:  
  - Core of **merge sort**, a parallelizable sorting algorithm (Chapter 13).  
  - Used in **map-reduce frameworks** (e.g., Hadoop) for assembling sorted results efficiently.  

#### **12.2 A Sequential Merge Algorithm**  
- **Algorithm Steps**:  
  1. **Merge Loop**: Sequentially compares elements from **A** and **B**, placing the smaller one into **C**.  
  2. **Remaining Elements**: After one array is exhausted, the remaining elements from the other are copied into **C**.  
- **Complexity**: **O(m + n)** (linear time, proportional to total elements).  
- **Example**:  
  - For **A = [3, 5, 7, 7, 10]** and **B = [1, 6, 7, 10]**, the algorithm merges them into **C = [1, 3, 5, 6, 7, 7, 7, 10, 10]**.  

#### **Key Takeaways**  
- Merge is fundamental for sorting and distributed computing.  
- The sequential algorithm is efficient (**O(m + n)**) but can be parallelized for better performance.  
- Stability ensures preservation of prior orderings when keys are equal.  

