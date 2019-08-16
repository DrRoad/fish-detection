import torch.nn as nn
import torch.nn.functional as F


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`,"
    "a module from fastai v3."
    def __init__(self, output_size=None):
        "Output will be 2*output_size or 2 if output_size is None"
        super().__init__()
        self.output_size = output_size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


class ResNetFish(nn.Module):
    def __init__(self, net, n_classes):
        super().__init__()
        adapt_pool = AdaptiveConcatPool2d((1,1))
        flatten = Flatten()
        fc = nn.Linear(in_features=2048*2, out_features=n_classes)
        nn.init.xavier_uniform_(fc.weight)
        nn.init.constant_(fc.bias, 0.)

        self.group_1 = nn.Sequential(*[c for c in net.children()][:5])
        self.group_2 = nn.Sequential(*[c for c in net.children()][5:8])
        self.group_3 = nn.Sequential(adapt_pool, flatten, fc)

    def forward(self, x):
        x = self.group_1(x)
        x = self.group_2(x)
        out = self.group_3(x)
        return out