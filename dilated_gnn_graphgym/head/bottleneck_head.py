import torch
import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.register import register_head


@register_head('bottleneck_head')
class BottleneckHead(nn.Module):
  
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out+1, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))
        

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = batch.x[batch.root_mask]
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label