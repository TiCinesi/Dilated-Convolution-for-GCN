import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.config import cfg

cfg.gnn.att_heads_final = 6
cfg.gnn.att_heads = 4
@register_layer('edge_gatconv')
class EGATConv(nn.Module):
    """
    Graph Attention Network (GAT) layer with edge features

    Note: In the original paper, they already present the support for edge attributes.
    """
    def __init__(self, layer_config: LayerConfig, num_heads, attention_concat, **kwargs):
        super().__init__()
        self.model = pyg.nn.GATConv(layer_config.dim_in, layer_config.dim_out, edge_dim=layer_config.edge_dim,
                                    bias=layer_config.has_bias, heads=num_heads, concat=attention_concat)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        return batch

@register_layer('gatconv_paper')
class GATConv(nn.Module):
    """
    Graph Attention Network (GAT) layer
    """
    def __init__(self, layer_config: LayerConfig, num_heads, attention_concat, **kwargs):
        super().__init__()
        self.model = pyg.nn.GATConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias, heads=num_heads, concat=attention_concat)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('gatv2conv')
class GATv2Conv(nn.Module):
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GATv2Conv(layer_config.dim_in, layer_config.dim_out, bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

@register_layer('edge_gatv2conv')
class EdgeGATv2Conv(nn.Module):
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GATv2Conv(layer_config.dim_in, layer_config.dim_out, edge_dim=layer_config.edge_dim,
                                    bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        return batch