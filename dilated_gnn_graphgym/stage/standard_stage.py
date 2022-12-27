import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage

from dilated_gnn_graphgym.layer.dilated_gnn_layer import create_dilated_gnn_layer
from torch_geometric.graphgym.models.layer import GeneralLayer, new_layer_config

def create_classic_gnn_layer(dim_in, dim_out, has_act=True):
    """
    Wrapper for a GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    """
    layer_conf = new_layer_config(dim_in, dim_out, 1, has_act=has_act,
                                      has_bias=False, cfg=cfg)
    return GeneralLayer(
        cfg.gnn.layer_type,
        layer_config=layer_conf)


cfg.gnn.layers_k1 = 1
cfg.gnn.layers_k2 = 1
cfg.gnn.act_on_last_layer_mp = True
@register_stage('standard_stage')
class GNNStackStage(nn.Module):
    """
    Simple Stage that stack GNN layers

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of GNN layers
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            has_act = True
            if i == num_layers - 1: #last layer
                has_act = cfg.gnn.act_on_last_layer_mp
            layer = create_classic_gnn_layer(d_in, dim_out, has_act)
            self.add_module('layer{}'.format(i), layer)

    def forward(self, batch):
        """"""
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif cfg.gnn.stage_type == 'skipconcat' and \
                    i < self.num_layers - 1:
                batch.x = torch.cat([x, batch.x], dim=1)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch
