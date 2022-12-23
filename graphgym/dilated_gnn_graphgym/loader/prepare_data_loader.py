from torch_geometric.graphgym.register import register_loader
from torch_geometric.graphgym.loader import load_pyg, load_ogb
from torch_geometric.graphgym.config import cfg

from dilated_gnn_graphgym.transform.transform_edge_connectivity import EdgeConnectivity

@register_loader('example')
def load_dataset_example(format, name, dataset_dir):
    if format == 'PyG':
        dataset = load_pyg(name, dataset_dir)
    elif format == 'OGB':
        dataset = load_ogb(name.replace('_', '-'), dataset_dir)
    else:
        raise ValueError('Unknown data format: {}'.format(cfg.dataset.format))

    t = EdgeConnectivity(cfg.gnn.layers_k1, cfg.gnn.layers_k2)
    dataset.transform = t #set transformer
    return dataset

 