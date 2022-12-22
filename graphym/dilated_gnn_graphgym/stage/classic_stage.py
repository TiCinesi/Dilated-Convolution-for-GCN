import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import GeneralLayer
from torch_geometric.graphgym.register import register_stage



@register_stage('classic-stage')
class GNNClassicStage(nn.Module):
    def __init__(self, k1, dim_in, hidden_dim, has_act, base_layer_name):
        super().__init__()
        self.k1 = k1
        for i in range(self.k1):
            d_in = dim_in if i == 0 else hidden_dim
            layer = GeneralLayer(base_layer_name, d_in, hidden_dim, has_act)
            self.add_module(f'layer{i}', layer)


    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch
