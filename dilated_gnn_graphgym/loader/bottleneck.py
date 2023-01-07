
import multiprocessing
from typing import Dict, List, Optional, Tuple
import torch
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

import numpy as np
import itertools
import random
import math

from torch_geometric.graphgym.config import cfg
from dilated_gnn_graphgym.loader.dataloader import set_dataset_info, get_loader

#Source: https://github.com/tech-srl/bottleneck/blob/bfe83b4a6dd7939ddb19cabea4f1e072f3c35432/tasks/tree_dataset.py
class TreeDataset(object):
    """
    @inproceedings{
        alon2021on,
        title={On the Bottleneck of Graph Neural Networks and its Practical Implications},
        author={Uri Alon and Eran Yahav},
        booktitle={International Conference on Learning Representations},
        year={2021},
        url={https://openreview.net/forum?id=i80OPhOCVH2}
    }
    """
    def __init__(self, depth):
        super(TreeDataset, self).__init__()
        self.depth = depth
        self.num_nodes, self.edges, self.leaf_indices = self._create_blank_tree()
        self.criterion = F.cross_entropy

    def add_child_edges(self, cur_node, max_node):
        edges = []
        leaf_indices = []
        stack = [(cur_node, max_node)]
        while len(stack) > 0:
            cur_node, max_node = stack.pop()
            if cur_node == max_node:
                leaf_indices.append(cur_node)
                continue
            left_child = cur_node + 1
            right_child = cur_node + 1 + ((max_node - cur_node) // 2)
            edges.append([left_child, cur_node])
            edges.append([right_child, cur_node])
            stack.append((right_child, max_node))
            stack.append((left_child, right_child - 1))
        return edges, leaf_indices

    def _create_blank_tree(self):
        max_node_id = 2 ** (self.depth + 1) - 2
        edges, leaf_indices = self.add_child_edges(cur_node=0, max_node=max_node_id)
        return max_node_id + 1, edges, leaf_indices

    def create_blank_tree(self, add_self_loops=True):
        edge_index = torch.tensor(self.edges).t()
        edge_index = torch_geometric.utils.to_undirected(edge_index) #NOTE debug 
        if add_self_loops:
            edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index=edge_index, )
        return edge_index

    def generate_data(self):
        data_list = []

        for comb in self.get_combinations():
            edge_index = self.create_blank_tree(add_self_loops=True)
            nodes = torch.tensor(self.get_nodes_features(comb), dtype=torch.long)
            root_mask = torch.tensor([True] + [False] * (len(nodes) - 1))
            label = self.label(comb)
            data_list.append(Data(x=nodes, edge_index=edge_index, root_mask=root_mask, y=label))

        dim0, out_dim = self.get_dims()
   
        return data_list, dim0, out_dim

    # Every sub-class should implement the following methods:
    def get_combinations(self):
        raise NotImplementedError

    def get_nodes_features(self, combination):
        raise NotImplementedError

    def label(self, combination):
        raise NotImplementedError

    def get_dims(self):
        raise NotImplementedError

# Source: https://github.com/tech-srl/bottleneck/blob/bfe83b4a6dd7939ddb19cabea4f1e072f3c35432/tasks/dictionary_lookup.py
class DictionaryLookupDataset(TreeDataset):
    """
    @inproceedings{
        alon2021on,
        title={On the Bottleneck of Graph Neural Networks and its Practical Implications},
        author={Uri Alon and Eran Yahav},
        booktitle={International Conference on Learning Representations},
        year={2021},
        url={https://openreview.net/forum?id=i80OPhOCVH2}
    }
    """
    def __init__(self, depth):
        super(DictionaryLookupDataset, self).__init__(depth)

    def get_combinations(self):
        # returns: an iterable of [key, permutation(leaves)]
        # number of combinations: (num_leaves!)*num_choices
        num_leaves = len(self.leaf_indices)
        num_permutations = 1000
        max_examples = 32000

        if self.depth > 3:
            per_depth_num_permutations = min(num_permutations, math.factorial(num_leaves), max_examples // num_leaves)
            permutations = [np.random.permutation(range(1, num_leaves + 1)) for _ in
                            range(per_depth_num_permutations)]
        else:
            permutations = random.sample(list(itertools.permutations(range(1, num_leaves + 1))),
                                         min(num_permutations, math.factorial(num_leaves)))

        return itertools.chain.from_iterable(

            zip(range(1, num_leaves + 1), itertools.repeat(perm))
            for perm in permutations)

    def get_nodes_features(self, combination):
        # combination: a list of indices
        # Each leaf contains a one-hot encoding of a key, and a one-hot encoding of the value
        # Every other node is empty, for now
        selected_key, values = combination

        # The root is [one-hot selected key] + [0 ... 0]
        nodes = [ (selected_key, 0) ]

        for i in range(1, self.num_nodes):
            if i in self.leaf_indices:
                leaf_num = self.leaf_indices.index(i)
                node = (leaf_num+1, values[leaf_num])
            else:
                node = (0, 0)
            nodes.append(node)
        return nodes

    def label(self, combination):
        selected_key, values = combination
        return int(values[selected_key - 1])

    def get_dims(self):
        # get input and output dims
        in_dim = len(self.leaf_indices)
        out_dim = len(self.leaf_indices)
        return in_dim, out_dim


from torch_geometric.data import Dataset
from torch_geometric.data.in_memory_dataset import nested_iter
from torch_geometric.data.separate import separate
from torch_geometric.data.collate import collate
import copy
from joblib import Parallel, delayed
from collections.abc import Mapping, Sequence
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
def nested_iter(node: Union[Mapping, Sequence]) -> Iterable:
    if isinstance(node, Mapping):
        for key, value in node.items():
            for inner_key, inner_value in nested_iter(value):
                yield inner_key, inner_value
    elif isinstance(node, Sequence):
        for i, inner_value in enumerate(node):
            yield i, inner_value
    else:
        yield None, node


class BottleneckDataset(Dataset):
    @staticmethod
    def collate(
            data_list: List[Data]) -> Tuple[Data, Optional[Dict[str, Tensor]]]:
        r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
        to the internal storage format of
        :class:`~torch_geometric.data.InMemoryDataset`."""
        if len(data_list) == 1:
            return data_list[0], None

        data, slices, _ = collate(
            data_list[0].__class__,
            data_list=data_list,
            increment=False,
            add_batch=False,
        )

        return data, slices

    def __init__(self, data_list, transform=None, pre_transform=None, pre_filter=None, split=None):
        super().__init__(None, transform, pre_transform, pre_filter)
        self.split = split
        self.original_len = len(data_list)
        if pre_transform is not None:
            # @ray.remote
            # def f(data, fn): 
            #     #data.edge_index = torch_geometric.utils.to_undirected(data.edge_index)
            #     return fn(data)
            
            # p = ray.put(self.pre_transform)
            # self.data_list = ray.get([f.remote(d,p) for d in data_list])

            data_list = Parallel(n_jobs=-1)(delayed(self.pre_transform)(d) for d in data_list)
            print('Done trasform for ', split)


        self._data_list: Optional[List[Data]] = None
        self.data, self.slices = self.collate(data_list)


    def len(self) -> int:
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            if self.split == 'train':
                return (len(value) - 1)*100
            else:
                return len(value) - 1
        return 0


    def get(self, idx: int) -> Data:
        if self.split == 'train':
            idx = idx % self.original_len
        
        if self.len() == 1:
            return copy.copy(self.data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        self._data_list[idx] = copy.copy(data)

        return data

cfg.share.dim0 = -1
def create_bottleneck_dataloader(name, dataset_dir, transform, pre_transform):
    #name depth_n
    n = int(name[6:])
    deep_dataset = DictionaryLookupDataset(n)
    data_list, cfg.share.dim0, out_dim = deep_dataset.generate_data()

   
    dataset_train, dataset_val = train_test_split(data_list, train_size=0.8, shuffle=True, stratify=[data.y for data in data_list])

    # dataset_train = dataset_train *1000
    dataset_train = BottleneckDataset(dataset_train, transform=transform, pre_transform=pre_transform, split='train')
    dataset_val = BottleneckDataset(dataset_val, transform=transform, pre_transform=pre_transform)
    
    set_dataset_info(dataset_train)
    cfg.share.dim_in = data_list[0].x.shape[1]
    cfg.share.dim_out  = out_dim
    cfg.share.num_splits  = 2
    
    print('BottleneckDataset preprocessed')
    train_loader =  get_loader(dataset_train, cfg.train.sampler, cfg.train.batch_size, shuffle=True)
    val_loader = get_loader(dataset_val, cfg.val.sampler, cfg.train.batch_size, shuffle=False)

    return train_loader,  val_loader

