import torch.nn as nn
import torch


class Ensemble(nn.Module):
    def __init__(self, create_net_fn, num_ensembles):
        super().__init__()
        self.nets = nn.ModuleList([create_net_fn() for _ in range(num_ensembles)])

    def forward(self, *argv):
        outs = []
        for net in self.nets:
            net_out = net(*argv)
            outs.append(net_out)

        if isinstance(outs[0], torch.Tensor):
            return torch.stack(outs)
        return outs
