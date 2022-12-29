
from typing import Optional, Union
from torch import Tensor
from torch_sparse import SparseTensor, matmul

from torch_geometric.nn.conv import GCNConv
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size

import torch
import torch.nn as nn

import torch_geometric as pyg
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.nn.inits import reset
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

def gcn_norm(edge_index, edge_weight=None, num_nodes=None,
            flow="source_to_target", dtype=None):

  
    assert flow in ["source_to_target", "target_to_source"]

   

   
    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class EdgeGCNConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, 
                 fill_value: Union[float, Tensor, str] = 'mean',
                 edge_agg: str = 'add',
                 edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(EdgeGCNConv, self).__init__( **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = Linear_pyg(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        self.edge_agg = edge_agg
        self.fill_value = fill_value

        if edge_dim is not None and edge_agg == 'add':
            self.map_edges = Linear_pyg(edge_dim, in_channels)
        else:
            self.map_edges = None

        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)
        if self.map_edges:
            self.map_edges.reset_parameters()

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        if self.add_self_loops:
            num_nodes = maybe_num_nodes(edge_index,  x.size(self.node_dim))

            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, self.fill_value, num_nodes)
    
        edge_index, edge_attr = gcn_norm(  # yapf: disable
                        edge_index, edge_attr, num_nodes,
                        self.flow, x.dtype)


        norm = self.norm(edge_index, x.size(0), x.dtype)

        out =  self.propagate(edge_index, x=x, edge_attr=edge_attr, norm = norm)
        
        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_attr, norm):
        if self.edge_agg == 'add':
            if self.map_edges is not None:
                edge_attr = self.map_edges(edge_attr)
            x_j = x_j + edge_attr
        elif self.edge_agg == 'concat':
            x_j = torch.cat([x_j , edge_attr], dim=1)
        
        return norm.view(-1, 1) * self.lin(x_j)

@register_layer('edge_gcnconv')
class LayerEdgeGCNConv(nn.Module):
    """
    Graph Convolutional Network (GCN) layer with edge attributes
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()

        if cfg.gnn.edge_agg == 'add':
            edge_dim = layer_config.edge_dim
            if edge_dim == layer_config.dim_in:
                edge_dim = None
        elif cfg.gnn.edge_agg == 'concat':
            layer_config.dim_in = layer_config.dim_in + layer_config.edge_dim
            edge_dim = None


        self.model = EdgeGCNConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias, edge_agg=cfg.gnn.edge_agg,
                                    edge_dim=edge_dim)

    def forward(self, batch):       
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch