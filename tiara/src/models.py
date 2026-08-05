import torch
from torch import nn
import torch.nn.functional as F


# 0 - plastid
# 1 - bacteria
# 2 - mitochondria
# 3 - archeal
# 4 - eukarya


def mish(x):
    return x * torch.tanh(F.softplus(x))


class mish_layer(nn.Module):
    def __init__(self):
        super(mish_layer, self).__init__()

    def forward(self, _input):
        return mish(_input)


def _layers(dim_in, hidden_1, hidden_2, dim_out, dropout):
    """Build the MLP stack, allowing hidden_2 to be absent.

    The HP search may pick a SINGLE hidden layer, and it does: v2.1.3 published
    ``second_k-6_hidden_1-128_hidden_2-none_...pkl``. The old two-layer-only
    constructor could not be instantiated for that architecture at all, so the
    published model was unloadable at inference time.
    """
    stack = [nn.Linear(dim_in, hidden_1), nn.Dropout(dropout), nn.ReLU()]
    last = hidden_1
    if hidden_2 is not None and str(hidden_2).lower() not in ("none", ""):
        hidden_2 = int(hidden_2)
        stack += [nn.Linear(hidden_1, hidden_2), nn.ReLU(), nn.Dropout(dropout)]
        last = hidden_2
    stack += [nn.Linear(last, dim_out), nn.Softmax(1)]
    return stack


class NNet1(nn.Sequential):
    def __init__(self, dim_in, hidden_1, hidden_2, dim_out, dropout):
        super().__init__(*_layers(dim_in, hidden_1, hidden_2, dim_out, dropout))


class NNet2(nn.Sequential):
    def __init__(self, dim_in, hidden_1, hidden_2, dim_out, dropout):
        super().__init__(*_layers(dim_in, hidden_1, hidden_2, dim_out, dropout))
