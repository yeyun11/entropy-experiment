import torch
from entcal import batch_histc

torch.manual_seed(0)
a = torch.randn(256)

histc = torch.histc(a, bins=16, min=-3, max=3).int()
bhistc = batch_histc(a[None, ], min_value=-3, max_value=3, num_bins=16)[0]

print(histc.tolist(), histc.sum().item())
print(bhistc.tolist(), bhistc.sum().item())
print((histc - bhistc).abs().sum().item())
