import copy
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric as pyg
import torch_geometric.graphgym.models.act
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.contrib.layer.generalconv import (
    GeneralConvLayer,
    GeneralEdgeConvLayer,
)
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.graphgym.models.layer import LayerConfig






#@register_layer('esageconv')
class ESAGEConv(nn.Module):
    """
    GraphSAGE Conv layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.SAGEConv(layer_config.dim_in, layer_config.dim_out,
                                     bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch





#@register_layer('generalconv')
class EGeneralConv(nn.Module):
    """A general GNN layer"""
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GeneralConvLayer(layer_config.dim_in,
                                      layer_config.dim_out,
                                      bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch
