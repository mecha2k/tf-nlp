import torch

sample = torch.randn(3, 5, 512)
print(sample.shape)
print(sample.transpose(2, 1).shape)
print(sample.transpose(1, 2).shape)
print(sample.transpose(2, 0).shape)
print(sample.transpose(0, 2).shape)
