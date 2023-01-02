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
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.embedding_list.append(emb)

    def forward(self, batch):
        """"""
        encoded_features = 0
        for i in range(batch.x.shape[1]):
            encoded_features += self.embedding_list[i](batch.x[:, i])

        batch.x = encoded_features
        return batch