
import torch
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_pooling
from torch_geometric.nn import (
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

@register_pooling('max_of_concat_layers')
def max_of_concat_layers(x_all, batch):
    h = []
    for i in range(x_all.shape[0]): #assume is list of layers
        h.append(x_all[i])
    x = torch.cat(h, dim=1)
    x = global_max_pool(x, batch)

    return x
