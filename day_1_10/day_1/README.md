# Using the CUDA Extension with PyTorch

This guide explains how to build and use a CUDA extension (`increment.cu`) with PyTorch.

## Prerequisites

- A Python environment with **PyTorch** installed, including **GPU capability**.
- NVIDIA CUDA Toolkit installed and configured properly.

## Steps to Build and Install the Extension

1. Navigate to the `day_1` folder:
   ```bash
   cd day_1
2. Build the extension:
    ```bash
    python setup.py build
3. Install the extension
    ```bash
    python setup.py install
# Using the Extension

Once the extension is installed, you can use it in your PyTorch code as demonstrated in the use_increment.py file. Here’s a quick example:
```python
import torch
import increment_extension

tensor = torch.randn(1024)
tensor = tensor * 4 + 5
tensor = tensor.to(torch.int32).to('cuda')

print('before increment', tensor[:10])
increment_extension.increment(tensor)
print('after increment', tensor[:10])
```
## Notes

- Ensure your PyTorch version is compatible with your CUDA installation.  
- If you encounter any issues with compilation, check your system's CUDA setup and verify that all dependencies are correctly installed.  
- If you are using a conda environment named `conda_env` and Python 3.11, make sure to add the following paths to your `includePath` (this is necessary to use `<torch/extension.h>`):  
```bash
/home/username/.conda/envs/conda_env/lib/python3.11/site-packages/torch/include
/home/username/.conda/envs/conda_env/lib/python3.11/site-packages/torch/include/torch/csrc/api/include