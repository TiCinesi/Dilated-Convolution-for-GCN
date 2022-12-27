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



@register_layer('eginconv')
class EGINConv(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer with edge features

    NOTE paper: https://arxiv.org/pdf/1905.12265.pdf
    https://github.com/sangyx/gtrick/blob/907742ac5761fab22508e5e5fc5186b4c6bde5ca/benchmark/pyg/model.py#L8
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = nn.Sequential(
            Linear_pyg(layer_config.dim_in, layer_config.dim_out), nn.ReLU(),
            Linear_pyg(layer_config.dim_out, layer_config.dim_out))
        self.model = pyg.nn.GINEConv(gin_nn, edge_dim=layer_config.edge_dim)

    def forward(self, batch):       
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch


@register_layer('egatconv')
class EGATConv(nn.Module):
    """
    Graph Attention Network (GAT) layer with edge features

    Note: In the original paper, they already present the support for edge attributes.
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GATConv(layer_config.dim_in, layer_config.dim_out, edge_dim=layer_config.edge_dim,
                                    bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        return batch


#@register_layer('egcnconv')
class EGCNConv(nn.Module):
    """
    Graph Convolutional Network (GCN) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg.nn.GCNConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


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
