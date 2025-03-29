import torch
import increment_extension

tensor = torch.randn(1024)
tensor = tensor * 4 + 5
tensor = tensor.to(torch.int32).to('cuda')

print('before increment', tensor[:10])
increment_extension.increment(tensor)
print('after increment', tensor[:10])
