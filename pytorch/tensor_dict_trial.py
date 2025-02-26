import torch
from tensordict import TensorDict

a = torch.randn(3, 4)
b = torch.randn(3, 4, 5)
td = TensorDict(dict(a=a, b=b), batch_size=[3, 4])

print(f"{td=}")
print(f"{td.select('a')=}")
print(f"{td.chunk(2, dim=1)=}")
