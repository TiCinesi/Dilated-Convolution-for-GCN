import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_stage

from dilated_gnn_graphgym.layer.dilated_gnn_layer import create_dilated_gnn_layer
from torch_geometric.graphgym.models.layer import GeneralLayer, new_layer_config

def create_classic_gnn_layer(dim_in, dim_out, has_act=True, final=False):
    """
    Wrapper for a GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    """
    layer_conf = new_layer_config(dim_in, dim_out, 1, has_act=has_act,
                                      has_bias=False, cfg=cfg)

    if  cfg.gnn.layer_type == 'edge_gatconv' or  cfg.gnn.layer_type == 'gatconv_paper':
        num_heads = cfg.gnn.att_heads_final if final else cfg.gnn.att_heads
        return GeneralLayer(
            cfg.gnn.layer_type,
            layer_config=layer_conf, num_heads=num_heads, attention_concat=not final)
    else:
        return GeneralLayer(
            cfg.gnn.layer_type,
            layer_config=layer_conf)

cfg.gnn.layers_k1 = 1
cfg.gnn.layers_k2 = 1
cfg.gnn.act_on_last_layer_mp = True
cfg.gnn.layer_norm = False
class GNNStandardStage(nn.Module):
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
        self.cfg = cfg
        self.layer_norms = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            has_act = True
            final = False
            if i == num_layers - 1: #last layer
                has_act = cfg.gnn.act_on_last_layer_mp
                final = True
            layer = create_classic_gnn_layer(d_in, dim_out, has_act, final)
            self.layers.append(layer)

            if cfg.gnn.layer_norm:
                self.layer_norms.append(nn.LayerNorm(dim_out))

    def forward(self, batch):
        """"""
        if self.cfg.model.graph_pooling == 'concat_across_sum_of_layers' or self.cfg.model.graph_pooling == 'max_of_concat_layers':
            h = []
            for i in range(self.num_layers):
                layer = self.layers[i]
                x = batch.x
                batch = layer(batch)
                if cfg.gnn.stage_type == 'skipsum':
                    batch.x = x + batch.x
                elif cfg.gnn.stage_type == 'skipconcat' and \
                        i < self.num_layers - 1:
                    batch.x = torch.cat([x, batch.x], dim=1)
                h.append(batch.x)
            batch.x = torch.stack(h)
            if self.cfg.gnn.l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=-1)
            return batch

        else:
            for i in range(self.num_layers):
                layer = self.layers[i]
                x = batch.x
                batch = layer(batch)
                if self.cfg.gnn.stage_type == 'skipsum':
                    batch.x = x + batch.x
                elif self.cfg.gnn.stage_type == 'skipconcat' and \
                        i < self.num_layers - 1:
                    batch.x = torch.cat([x, batch.x], dim=1)
            if self.cfg.gnn.l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=-1)
            if self.cfg.gnn.layer_norm:
                batch.x = self.layer_norms[i](batch.x)
            return batch
