import torch
from torch_geometric.graphgym.register import register_loader
from torch_geometric.graphgym.loader import load_pyg, load_ogb
from torch_geometric.graphgym.config import cfg

from dilated_gnn_graphgym.transform.transform_edge_connectivity import EdgeConnectivity
from dilated_gnn_graphgym.transform.transform_edge_connectivity_features import EdgeConnectivityAndFeatures


cfg.gnn.use_edge_features = False
@register_loader('load_dataset-trasform') #dummy loader, didn't find where to specify which loader
def load_dataset(format, name, dataset_dir):

    if cfg.gnn.use_edge_features:
        t = EdgeConnectivityAndFeatures(cfg.gnn.layers_k1, cfg.gnn.layers_k2)
    else:
        t = EdgeConnectivity(cfg.gnn.layers_k1, cfg.gnn.layers_k2)

    if format == 'PyG':
        dataset = load_pyg(name, dataset_dir)
    elif format == 'OGB':
        dataset = load_ogb(name.replace('_', '-'), dataset_dir)
    else:
        raise ValueError('Unknown data format: {}'.format(cfg.dataset.format))
    dataset.transform = t #set transformer
    return dataset

 