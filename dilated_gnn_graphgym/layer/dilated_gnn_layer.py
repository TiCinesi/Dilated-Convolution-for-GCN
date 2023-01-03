import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import GeneralLayer, LayerConfig, new_layer_config
import torch_geometric.graphgym.register as register


def create_dilated_gnn_layer(dim_in, dim_out, has_act=True, final=False):
    """
    Wrapper for a Dilated GNN layer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        has_act (bool): Whether has activation function after the layer

    """
    layer_conf = new_layer_config(dim_in, dim_out, 1, has_act=has_act,
                                      has_bias=False, cfg=cfg)
    
    if  cfg.gnn.layer_type == 'edge_gatconv' or  cfg.gnn.layer_type == 'gatconv_paper':
        num_heads = cfg.gnn.att_heads_final if final else cfg.gnn.att_heads
        
        return DilatedGeneralLayer(
            cfg.gnn.layer_type,
            layer_config=layer_conf, num_heads=num_heads, attention_concat=not final)
    else:
        return DilatedGeneralLayer(
            cfg.gnn.layer_type,
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

cfg.gnn.use_edge_features = False
class DilatedGeneralLayer(GeneralLayer):
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


    def forward(self, batch, step):
        edge_index = batch.__getattr__(f'dilated_step_{step}_edge_index')
        
        # NOTE insted of doing self.layer(batch), we access self.layer.model, 
        # to be able to choose which set of edges we are using, and which edge attributes.
        if cfg.gnn.use_edge_features:
            ids = batch.__getattr__(f'dilated_step_{step}_path_ids')
            edge_feature = batch.edge_attr[ids]
            edge_feature = self.gem(edge_feature)

            batch.x = self.layer.model(x=batch.x, edge_index=edge_index, edge_attr=edge_feature)
        else:
            batch.x = self.layer.model(batch.x, edge_index)
       
        batch.x = self.post_layer(batch.x)
        if self.has_l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=1)
        
        return batch
