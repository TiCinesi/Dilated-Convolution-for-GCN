import torch
from torch_geometric.graphgym.register import register_loader
from dilated_gnn_graphgym.loader.loader import load_pyg, load_ogb
from torch_geometric.graphgym.config import cfg

from dilated_gnn_graphgym.transform.transform_edge_connectivity import EdgeConnectivity
from dilated_gnn_graphgym.transform.transform_edge_connectivity_features import EdgeConnectivityAndFeatures

from dilated_gnn_graphgym.transform.transform_node_features import EmptyNodeFeatures 

cfg.gnn.use_edge_features = False
cfg.dataset.fast_loader = False
@register_loader('load_dataset_trasform') #dummy loader, didn't find where to specify which loader
def load_dataset(format, name, dataset_dir):

    if cfg.gnn.stage_type == 'dilated_stage':
        if cfg.gnn.use_edge_features :
            t = EdgeConnectivityAndFeatures(cfg.gnn.layers_k1, cfg.gnn.layers_k2)
        else:
            t = EdgeConnectivity(cfg.gnn.layers_k1, cfg.gnn.layers_k2)
    else:
        t = EmptyNodeFeatures()

    if format == 'PyG':
        dataset = load_pyg(name, dataset_dir)
    elif format == 'OGB':
        dataset = load_ogb(name.replace('_', '-'), dataset_dir)
    else:
        raise ValueError('Unknown data format: {}'.format(cfg.dataset.format))
    dataset.transform = t #set transformer

    if dataset.data.edge_attr is not None:
        # Auto set edge_dim based on dataset. It will be auto-updated if edge encoding is performed.
        cfg.dataset.edge_dim = dataset.data.edge_attr.shape[1]

    return dataset

 