import torch
from torch import Tensor
import torch.nn as nn
import math
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import GeneralLayer, LayerConfig, new_layer_config
import torch_geometric.graphgym.register as register

cfg.gnn.layer_type_dilated = 'edge_ginconv'
def create_dilated_positional_gnn_layer(dim_in, dim_out, has_act=True, final=False):
    """
    Wrapper for a Dilated GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    """
    layer_conf = new_layer_config(dim_in, dim_out, 1, has_act=has_act,
                                      has_bias=False, cfg=cfg)
    
    if not cfg.gnn.use_edge_features:
        layer_conf.edge_dim = 2
    
    if  cfg.gnn.layer_type_dilated == 'edge_gatconv' or  cfg.gnn.layer_type_dilated == 'gatconv_paper':
        num_heads = cfg.gnn.att_heads_final if final else cfg.gnn.att_heads
        return DilatedPositionalGeneralLayer(
            cfg.gnn.layer_type_dilated,
            layer_config=layer_conf, num_heads=num_heads, attention_concat=not final)
    else:
        return DilatedPositionalGeneralLayer(
            cfg.gnn.layer_type_dilated,
            layer_config=layer_conf)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        #NOTE I saw that they are clamping for stability
        return torch.mean(x.clamp(min=eps).pow(p), dim=1).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
      
    def forward(self, x: Tensor,  path_len: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [num_edges, d_model]
            path_len: [num_edges]
        """
        path_len = path_len
    
        pe = torch.zeros_like(x, device=x.device)

        for i in range(self.d_model):
            if i % 2 == 0:
                div_term = torch.exp(torch.tensor(i* (-math.log(10000.0) / self.d_model), device=x.device))
                pe[:, i] = torch.sin(path_len * div_term)
            else:
                div_term = torch.exp(torch.tensor((i-1)* (-math.log(10000.0) / self.d_model), device=x.device))
                pe[:, i] = torch.cos(path_len * div_term)
     
        x = x + pe
        return x

cfg.gnn.use_edge_features = False
class DilatedPositionalGeneralLayer(GeneralLayer):
    """
    General wrapper for  dilated layers layers

    Args:
        name (string): Name of the layer in registered :obj:`layer_dict`
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        layer_config (Layerconfig): Layer Config object
    """

    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__(name, layer_config, **kwargs)

        self.layer = register.layer_dict[name](layer_config, **kwargs)
        self.gem = GeM()
        self.cfg = cfg
        if cfg.gnn.use_edge_features:
            self.pos = PositionalEncoding(cfg.dataset.edge_dim)
        else:
            self.pos = PositionalEncoding(2)


    def forward(self, batch, step):
        if batch.edge_index is not None:
            edge_index = batch.__getattr__(f'dilated_step_{step}_edge_index')
        else:
            adj_t = batch.__getattr__(f'dilated_step_{step}_adj_t')
        
        if self.cfg.dataset.positional_encoding_path:
            path_len = batch.__getattr__(f'dilated_step_{step}_path_len')
            node_disconnected = batch.__getattr__(f'dilated_step_{step}_node_disconnected')
            
            # NOTE insted of doing self.layer(batch), we access self.layer.model, 
            # to be able to choose which set of edges we are using, and which edge attributes.
            if self.cfg.gnn.use_edge_features:
                ids = batch.__getattr__(f'dilated_step_{step}_path_ids')
                edge_feature = batch.edge_attr[ids]
                edge_feature = self.gem(edge_feature)
                # add positional ecoding
                edge_feature = self.pos(edge_feature, path_len)
            else:
                edge_feature = torch.zeros((edge_index.shape[1], 2), device=edge_index.device)
                # add positional ecoding
                edge_feature = self.pos(edge_feature, path_len)

            x = self.layer.model(x=batch.x, edge_index=edge_index, edge_attr=edge_feature)
            #set previous feature if no neighbors of node
        
            #batch.x = torch.where(node_disconnected.unsqueeze(1), batch.x, x)
            batch.x  = x
        
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
            
            return batch
        else:

            if self.cfg.gnn.use_edge_features:
                ids = batch.__getattr__(f'dilated_step_{step}_path_ids')
                edge_feature = batch.edge_attr[ids]
                edge_feature = self.gem(edge_feature)
                batch.x = self.layer.model(x=batch.x, edge_index=edge_index, edge_attr=edge_feature)
            else:
                if batch.edge_index is not None:
                    batch.x = self.layer.model(x=batch.x, edge_index=edge_index)
                else:
                    batch.x = self.layer.model(x=batch.x, edge_index=adj_t.t())

            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
            
            return batch