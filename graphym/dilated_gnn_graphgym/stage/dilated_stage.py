import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNLayer
from torch_geometric.graphgym.register import register_stage


cfg.gnn.layers_k1 = 1
cfg.gnn.layers_k2 = 1




@register_stage('dilated-stage')
class GNNDilatedStage(nn.Module):

    def __init__(self, dim_in, dim_out, num_layers) -> None:
        super().__init__()
        #Standard GNN
        self.k1 = cfg.gnn.layers_k1
        self.classic_layers = nn.ModuleList()
        for i in range(self.k1):
            d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.classic_layers.append(layer)

        # Dilated GNN
        self.k2 = cfg.gnn.layers_k2
        self.alphas = nn.Parameter(torch.full((self.k2,), 0.5))
        self.dilated_layers = nn.ModuleList()

        for i in range(self.k2):
            layer = GNNLayer(dim_out, dim_out)
            self.dilated_layers.append(layer)

    
    def forward(self, batch):
       
        #Classic layers
        for i in range(self.k1):
            batch = self.classic_layers[i](batch)

        x, edge_index = batch.x, batch.edge_index

        #Call GNN layers + skip connections
        for i in range(self.k2):
            layer = self.dilated_layers[i]
            batch.edge_index = batch.__getattr__(f'distance_graphs_{i}_edge_index')
            new_batch = layer(batch)

            new_batch.x = self.alphas[i]*new_batch.x + (torch.tensor(1.0) - self.alphas[i])*batch.x
            batch = new_batch

        batch.edge_index = edge_index

        batch.x = torch.cat([batch.x, x], dim=1)
        return batch       