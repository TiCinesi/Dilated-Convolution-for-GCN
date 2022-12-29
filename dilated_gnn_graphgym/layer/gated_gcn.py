import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.graphgym.models.layer import LayerConfig



#@register_layer('edge_gatconv')
class EGATConv(nn.Module):
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GatedGraphConv(layer_config.dim_in, layer_config.dim_out, edge_dim=layer_config.edge_dim,
                                    bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        return batch