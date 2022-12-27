import torch
from torch_geometric.graphgym.register import register_loader
from torch_geometric.graphgym.loader import load_pyg
from torch_geometric.graphgym.config import cfg

from dilated_gnn_graphgym.transform.transform_edge_connectivity import EdgeConnectivity
from dilated_gnn_graphgym.transform.transform_edge_connectivity_features import EdgeConnectivityAndFeatures
from torch_geometric.graphgym.models.transform import (
    create_link_label,
    neg_sampling_transform,
)
from torch_geometric.utils import (
    index_to_mask,
    negative_sampling,
    to_undirected,
)

from torch_geometric.graphgym.loader import set_dataset_attr

from dilated_gnn_graphgym.transform.transform_edge_connectivity_features_fast import EdgeConnectivityAndFeaturesFast
from dilated_gnn_graphgym.transform.transform_node_features import EmptyNodeFeatures 

def load_ogb(name, dataset_dir):
    r"""

    Load OGB dataset objects.


    Args:
        name (string): dataset name
        dataset_dir (string): data directory

    Returns: PyG dataset object

    """
    from ogb.graphproppred import PygGraphPropPredDataset
    from ogb.linkproppred import PygLinkPropPredDataset
    from ogb.nodeproppred import PygNodePropPredDataset

    if name[:4] == 'ogbn':
        dataset = PygNodePropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = ['train_mask', 'val_mask', 'test_mask']
        for i, key in enumerate(splits.keys()):
            mask = index_to_mask(splits[key], size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_names[i], mask, len(mask))
        #edge_index = to_undirected(dataset.data.edge_index) #NOTE don't convert to undirected, breaks stuff
        #set_dataset_attr(dataset, 'edge_index', edge_index,
        #                edge_index.shape[1])

    elif name[:4] == 'ogbg':
        dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_idx_split()
        split_names = [
            'train_graph_index', 'val_graph_index', 'test_graph_index'
        ]
        for i, key in enumerate(splits.keys()):
            id = splits[key]
            set_dataset_attr(dataset, split_names[i], id, len(id))

    elif name[:4] == "ogbl":
        dataset = PygLinkPropPredDataset(name=name, root=dataset_dir)
        splits = dataset.get_edge_split()
        id = splits['train']['edge'].T
        if cfg.dataset.resample_negative:
            set_dataset_attr(dataset, 'train_pos_edge_index', id, id.shape[1])
            dataset.transform = neg_sampling_transform
        else:
            id_neg = negative_sampling(edge_index=id,
                                       num_nodes=dataset.data.num_nodes,
                                       num_neg_samples=id.shape[1])
            id_all = torch.cat([id, id_neg], dim=-1)
            label = create_link_label(id, id_neg)
            set_dataset_attr(dataset, 'train_edge_index', id_all,
                             id_all.shape[1])
            set_dataset_attr(dataset, 'train_edge_label', label, len(label))

        id, id_neg = splits['valid']['edge'].T, splits['valid']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'val_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'val_edge_label', label, len(label))

        id, id_neg = splits['test']['edge'].T, splits['test']['edge_neg'].T
        id_all = torch.cat([id, id_neg], dim=-1)
        label = create_link_label(id, id_neg)
        set_dataset_attr(dataset, 'test_edge_index', id_all, id_all.shape[1])
        set_dataset_attr(dataset, 'test_edge_label', label, len(label))

    else:
        raise ValueError('OGB dataset: {} non-exist')
    return dataset

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

 