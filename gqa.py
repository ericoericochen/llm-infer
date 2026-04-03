import torch


a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

print(a)


b = torch.repeat_interleave(a, 2, dim=0)

print(b)
