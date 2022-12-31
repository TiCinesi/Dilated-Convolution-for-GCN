import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.init import init_weights
from torch_geometric.graphgym.models.layer import (
    BatchNorm1dNode,
    GeneralLayer,
    GeneralMultiLayer,
    new_layer_config,
)
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network


from dilated_gnn_graphgym.encoder.feature_encoder import FeatureEncoder
from dilated_gnn_graphgym.stage.dilated_stage import GNNDilatedStage

@register_network('dilated_gnn')
class DilatedGNNModel(nn.Module):
    """
    Custom GNN model: encoder + stage + head

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        **kwargs (optional): Optional additional args
    """
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        GNNHead = register.head_dict[cfg.gnn.head]

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner,
                                   cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        if cfg.gnn.layers_mp > 0:
            self.mp = GNNDilatedStage(dim_in=dim_in, dim_out=cfg.gnn.dim_inner,
                               num_layers=cfg.gnn.layers_mp)
        
        # NOTE here is the difference, since we are concatenating the computed features
        dim_in = cfg.gnn.dim_inner
        if cfg.gnn.dilated_path_join == 'concat':
            dim_in = 2*dim_in
        
        if cfg.gnn.dilated_path_join == 'concat' and  (cfg.model.graph_pooling == 'concat_across_sum_of_layers' or cfg.model.graph_pooling == 'max_of_concat_layers'):
            raise ValueError(f"Cannot activate both cfg.gnn.dilated_path_join={cfg.gnn.dilated_path_join}, and cfg.model.graph_pooling={cfg.model.graph_pooling}")

        if cfg.model.graph_pooling == 'concat_across_sum_of_layers' or cfg.model.graph_pooling == 'max_of_concat_layers':
            dim_in = dim_in * (cfg.gnn.layers_k1 + cfg.gnn.layers_k2)

        self.post_mp = GNNHead(dim_in=dim_in, dim_out=dim_out)


        self.apply(init_weights)

    def forward(self, batch):
        """"""
        for module in self.children():
            batch = module(batch)
        return batch
