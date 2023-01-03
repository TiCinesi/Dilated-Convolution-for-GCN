import torch

from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)

from torch_geometric.graphgym.config import cfg

@register_node_encoder('bottleneck_encoder')
class BottleneckEncoder(torch.nn.Module):
   
    def __init__(self, emb_dim):
        super().__init__()

        self.embedding_list = torch.nn.ModuleList()
        dim0 = cfg.share.dim0
        for i in range(2):
            emb = torch.nn.Embedding(dim0+1, emb_dim)
            self.embedding_list.append(emb)

    def forward(self, batch):
        """"""
        batch.x = self.embedding_list[0](batch.x[:, 0]) +  self.embedding_list[1](batch.x[:, 1])
        return batch