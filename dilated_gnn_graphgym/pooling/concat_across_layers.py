
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

@register_pooling('concat_across_sum_of_layers')
def concat_across_sum_of_layers(x, batch):
    h = []
    for i in range(x.shape[0]): #assume is list of layers
        h.append(global_add_pool(x[i], batch))
    
    return torch.cat(h, dim=1)
