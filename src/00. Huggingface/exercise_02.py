import torch
import torch.nn as nn
from icecream import ic


sample = torch.randn(3, 5, 512)
ic(sample.shape)
ic(sample.transpose(2, 1).shape)
ic(sample.transpose(1, 2).shape)
ic(sample.transpose(2, 0).shape)
ic(sample.transpose(0, 2).shape)


output = torch.Tensor(
    [
        [0.8982, 0.805, 0.6393, 0.9983, 0.5731, 0.0469, 0.556, 0.1476, 0.8404, 0.5544],
        [0.9457, 0.0195, 0.9846, 0.3231, 0.1605, 0.3143, 0.9508, 0.2762, 0.7276, 0.4332],
    ]
)
target = torch.LongTensor([1, 5])
ic(output.shape)
ic(target.shape)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output, target)
ic(loss)

output = output.unsqueeze(dim=0)
target = target.unsqueeze(dim=0)
ic(output.shape)
ic(target.shape)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(output.transpose(1, 2), target)
ic(loss)

x, y = 3, 7
inputs = torch.randn(size=(x, y))
labels = torch.randint(high=y, size=(x,))
labels2 = torch.randint(high=x, size=(y,))
ic(inputs.shape)
ic(labels.shape)
ic(labels2.shape)
ic(labels)
ic(labels2)
ic(loss_fn(inputs, labels))
ic(loss_fn(inputs.T, labels2))

batch_size = 2
num_len = 8
num_classes = 5
inputs = torch.randn(size=(batch_size, num_len, num_classes))
labels = torch.randint(low=0, high=num_classes, size=(batch_size, num_len))
ic(inputs.shape)
ic(labels.shape)
loss = loss_fn(inputs.transpose(1, 2), labels)
ic(loss)

loss1 = loss_fn(inputs[0], labels[0])
ic(loss1)
loss2 = loss_fn(inputs[1], labels[1])
ic(loss2)
loss = (loss1 + loss2) / 2
ic(loss)