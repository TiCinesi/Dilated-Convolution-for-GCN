import torch
import torch.nn as nn

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

@register_stage('dilated_stage')
class GNNDilatedEdgesFeatureStage(nn.Module):

    def __init__(self, dim_in, dim_out, num_layers) -> None:
        super().__init__()
        #Standard GNN
        self.k1 = cfg.gnn.layers_k1
        self.classic_layers = nn.ModuleList()
        for i in range(self.k1):
            d_in = dim_in if i == 0 else dim_out
            layer = create_classic_gnn_layer(d_in, dim_out)
            self.classic_layers.append(layer)

        # Dilated GNN
        self.k2 = cfg.gnn.layers_k2
        self.alphas = nn.Parameter(torch.full((self.k2,), 0.5))
        self.dilated_layers = nn.ModuleList()

        for i in range(self.k2):
            layer = create_dilated_gnn_layer(dim_out, dim_out)
            self.dilated_layers.append(layer)

    
    def forward(self, batch):
        #Classic layers
        for i in range(self.k1):
            batch = self.classic_layers[i](batch)

        x = batch.x

        #Call GNN layers + weighted residual connections (check terminology)
        for step in range(self.k2):
            layer = self.dilated_layers[step]
            new_batch = layer(batch, step)
            new_batch.x = self.alphas[step]*new_batch.x + (torch.tensor(1.0) - self.alphas[step])*batch.x
            batch = new_batch

        batch.x = torch.cat([batch.x, x], dim=1)
        return batch       