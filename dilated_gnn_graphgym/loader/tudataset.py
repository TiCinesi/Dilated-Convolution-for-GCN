import torch

import torch_geometric.graphgym.register as register

from torch_geometric.graphgym.config import cfg
from torch_geometric.datasets.tu_dataset import TUDataset

from dilated_gnn_graphgym.loader.dataloader import get_loader, set_dataset_info
from torch.utils.data import random_split

def create_loader_tu(name, dataset_dir, transform):
    
    dataset =  TUDataset(name=name, root=dataset_dir, transform=transform,
                        use_edge_attr=True, use_node_attr=True, cleaned=False)              
    set_dataset_info(dataset)
    cfg.share.num_splits  = 3
    dataset_train, dataset_val, dataset_test = random_split(dataset, cfg.dataset.split , generator=torch.Generator().manual_seed(cfg.seed)) #so it's round based

    train_loader =  get_loader(dataset_train, cfg.train.sampler, cfg.train.batch_size, shuffle=True)
    val_loader = get_loader(dataset_val, cfg.val.sampler, cfg.train.batch_size, shuffle=False)
    test_loader = get_loader(dataset_test, cfg.val.sampler, cfg.train.batch_size, shuffle=False)
    
    return train_loader,  val_loader, test_loader



