
from typing import Callable, Optional, Union
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

import torch
import torch.nn as nn

import torch_geometric as pyg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.nn.inits import reset

from torch_geometric.graphgym.config import cfg


class EdgeGINConv(MessagePassing):
    """
    GINConv with edge attributes
    """

    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        

        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            
            self.lin = Linear_pyg(edge_dim, in_channels)

        else:
            self.lin = None

        self.reset_parameters()
    
    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'EdgeGINVConv' or change 'edge_agg' to 'concat' ")
        
        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        return x_j + edge_attr

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn},lin={self.lin})'


def get_gin_nn(layer_config):
    gin_nn = [Linear_pyg(layer_config.dim_in, layer_config.dim_out)]
    if layer_config.has_batchnorm:
        gin_nn.append(nn.BatchNorm1d(layer_config.dim_out, eps=layer_config.bn_eps, momentum=layer_config.bn_mom))
    gin_nn.append(nn.ReLU())
    gin_nn.append(Linear_pyg(layer_config.dim_out, layer_config.dim_out))
    gin_nn = nn.Sequential(*gin_nn)
    return gin_nn

cfg.gnn.edge_agg = 'add'
@register_layer('edge_ginconv')
class LayerEdgeGINConv(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer with edge features

    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()

        edge_dim = layer_config.edge_dim
        if edge_dim == layer_config.dim_in:
            edge_dim = None

        gin_nn = get_gin_nn(layer_config)
        
        self.model = EdgeGINConv(gin_nn, edge_dim=edge_dim)

    def forward(self, batch):       
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch

@register_layer('ginconv_paper')
class LayerGINConv(nn.Module):
    """
    Graph Isomorphism Network (GIN) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = get_gin_nn(layer_config)
        self.model = pyg.nn.GINConv(gin_nn)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


@register_layer('edge_gineconv')
class EdgeGINEConv(nn.Module):
    """
    NOTE paper: https://arxiv.org/pdf/1905.12265.pdf
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()

        gin_nn = get_gin_nn(layer_config)

        edge_dim = layer_config.edge_dim
        if edge_dim == layer_config.dim_in:
            edge_dim = None

        self.model = pyg.nn.GINEConv(gin_nn, edge_dim=edge_dim)

    def forward(self, batch):       
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch