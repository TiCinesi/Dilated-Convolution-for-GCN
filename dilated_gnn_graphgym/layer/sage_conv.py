from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LSTM
from torch_sparse import SparseTensor, matmul
import torch.nn as nn
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.config import cfg

class EdgeSAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        edge_agg: str = 'add',
        edge_dim: Optional[int] = None,
        **kwargs):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project
        self.edge_agg = edge_agg

        if isinstance(in_channels, int):
            if edge_agg == 'concat' and edge_dim is not None:
                in_channels = (in_channels, in_channels - edge_dim)
            else:
                in_channels = (in_channels, in_channels)

        super().__init__(aggr, **kwargs)

        if self.project:
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        aggr_out_channels = in_channels[0]

        if edge_dim is not None and edge_agg == 'add':
            self.map_edges = Linear(edge_dim, in_channels[0])
        else:
            self.map_edges = None
        

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()
        if self.map_edges:
            self.map_edges.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: Tensor,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.edge_agg == 'add':
            if self.map_edges is not None:
                edge_attr = self.map_edges(edge_attr)
            x_j = x_j + edge_attr
        elif self.edge_agg == 'concat':
            x_j = torch.cat([x_j , edge_attr], dim=1)

        return x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


@register_layer('edge_sageconv')
class LayerEdgeSAGEConv(nn.Module):
    """
    Sage Conv layer with edge attributes
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()

        if cfg.gnn.edge_agg == 'add':
            edge_dim = layer_config.edge_dim
            if edge_dim == layer_config.dim_in:
                edge_dim = None
        elif cfg.gnn.edge_agg == 'concat':
            layer_config.dim_in = layer_config.dim_in + layer_config.edge_dim
            edge_dim = layer_config.edge_dim


        self.model = EdgeSAGEConv(layer_config.dim_in, layer_config.dim_out,
                                    bias=layer_config.has_bias, edge_agg=cfg.gnn.edge_agg,
                                    edge_dim=edge_dim)

    def forward(self, batch):       
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch
