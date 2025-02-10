import torch

import triton
import triton.language as tl

DEVICE = torch.device('cuda:0')

@triton.jit
def add_kernel(x_ptr, # Pointer to first input vector
                y_ptr, # Pointer to second input vector
                output_ptr, # Pointer to output vector
                n_elements, # size of the vector
                BLOCK_SIZE: tl.constexpr, # Number of elements each program should process
                ):
    
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x, device=DEVICE)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

if __name__ == '__main__':
    torch.manual_seed(42)
    size = 98432
    x = torch.randn(size=(size, ), device=DEVICE)
    y = torch.randn(size=(size, ), device=DEVICE)
    out_torch = x + y
    out_triton = add(x, y)
    print(out_torch, out_triton)

    print(f"The maxmimum difference between torch and triton is ", 
        f'{torch.max(torch.abs(out_torch - out_triton))}')

