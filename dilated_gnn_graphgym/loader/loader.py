import torch_geometric.transforms as T
from torch_geometric.datasets import (
    PPI,
    Amazon,
    Coauthor,
    KarateClub,
    MNISTSuperpixels,
    Planetoid,
    QM7b,
    TUDataset,
)
from torch_geometric.graphgym.config import cfg

from dilated_gnn_graphgym.loader.ogb import create_loader_ogb
from dilated_gnn_graphgym.loader.tudataset import create_loader_tu

from dilated_gnn_graphgym.transform.transform_edge_connectivity import EdgeConnectivity
from dilated_gnn_graphgym.transform.transform_edge_connectivity_features import EdgeConnectivityAndFeatures
from dilated_gnn_graphgym.transform.transform_node_features import EmptyNodeFeatures 


def ___load_pyg(name, dataset_dir):
    """
    Load PyG dataset objects. (More PyG datasets will be supported)

    Args:
        name (string): dataset name
        dataset_dir (string): data directory

    Returns: PyG dataset object

    """
    dataset_dir = '{}/{}'.format(dataset_dir, name)
    if name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(dataset_dir, name)
    elif name[:3] == 'TU_':
        # TU_IMDB doesn't have node features
        if name[3:] == 'IMDB':
            name = 'IMDB-MULTI'
            dataset = TUDataset(dataset_dir, name, transform=T.Constant())
        else:
            dataset = TUDataset(dataset_dir, name[3:])
    elif name == 'Karate':
        dataset = KarateClub()
    elif 'Coauthor' in name:
        if 'CS' in name:
            dataset = Coauthor(dataset_dir, name='CS')
        else:
            dataset = Coauthor(dataset_dir, name='Physics')
    elif 'Amazon' in name:
        if 'Computers' in name:
            dataset = Amazon(dataset_dir, name='Computers')
        else:
            dataset = Amazon(dataset_dir, name='Photo')
    elif name == 'MNIST':
        dataset = MNISTSuperpixels(dataset_dir)
    elif name == 'PPI':
        dataset = PPI(dataset_dir)
    elif name == 'QM7b':
        dataset = QM7b(dataset_dir)
    else:
        raise ValueError('{} not support'.format(name))

    return dataset


cfg.gnn.use_edge_features = False
def create_loader():
    """
    Create data loader objects in array
    [train, val, test]
    """

    format = cfg.dataset.format
    name = cfg.dataset.name
    dataset_dir = cfg.dataset.dir

    # Transformer based on dataset and model
    if cfg.model.type == 'dilated_gnn':
        if cfg.gnn.use_edge_features :
            transform = EdgeConnectivityAndFeatures(cfg.gnn.layers_k1, cfg.gnn.layers_k2)
        else:
            transform = EdgeConnectivity(cfg.gnn.layers_k1, cfg.gnn.layers_k2)
    else:
        transform = EmptyNodeFeatures()

    # Load from Pytorch Geometric dataset
    if format == 'TU':
        dataset = create_loader_tu(name, dataset_dir, transform)
    # Load from OGB formatted data
    elif format == 'OGB':
        dataset = create_loader_ogb(name, dataset_dir, transform)
    else:
        raise ValueError('Unknown data format: {}'.format(format))

    if cfg.gnn.use_edge_features and cfg.dataset.edge_dim == 0:
        raise ValueError('Edge feature are not present, but "cfg.gnn.use_edge_features" is True')

    return dataset




