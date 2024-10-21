import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import set_seed

dataset = TensorDataset(torch.arange(64))

set_seed(12345)
print("=" * 50, "using seed", "=" * 50)
loader = DataLoader(dataset, num_workers=2, shuffle=True)
print([x.item() for x, in loader])

print("=" * 50, "using seed", "=" * 50)
set_seed(12345)
loader = DataLoader(dataset, num_workers=2, shuffle=True)
print([x.item() for x, in loader])

print("=" * 50, "using generator", "=" * 50)
generator = torch.Generator().manual_seed(12345)
loader = DataLoader(dataset, num_workers=2, shuffle=True, generator=generator)
print([x.item() for x, in loader])

print("=" * 50, "using generator", "=" * 50)
generator = torch.Generator().manual_seed(12345)
loader = DataLoader(dataset, num_workers=2, shuffle=True, generator=generator)
print([x.item() for x, in loader])
