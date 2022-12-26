import torch
from torch_geometric.transforms import BaseTransform

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import math 
import random
from torch_geometric.typing import (
    OptTensor,
)

class DilatedInfoData(Data):

    def __init__(self, x: OptTensor = None, edge_index: OptTensor = None, edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if 'path_ids' in key:
            return self.num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'path_ids' in key:
            return 0
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

class EdgeConnectivityAndFeatures(BaseTransform):
    def __init__(self, k1, k2) -> None:
        super().__init__()
        self.k1 = k1 
        self.k2 = k2

    def __call__(self, data: Data) -> Data:
        
        if data.x is None:
            data.x = torch.zeros(data.num_nodes, dtype=torch.long)
        def random_neighbours(l):
            a = list(l)
            random.shuffle(a)
            return iter(a)


        if data.is_directed():
            raise ValueError("Graph is directed, cannot proceed!")
        
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)


        distance_graphs = []
        distance_graphs_attr_ids = []
        for i in range(self.k2):
            distance_graphs.append(nx.Graph())
            distance_graphs_attr_ids.append([])

        edge_ids = {}
        for i in range(data.edge_index.shape[1]):
            e = tuple(data.edge_index[:, i].numpy())
            edge_ids[e] = i #TODO check here


        for v in range(G.number_of_nodes()):
            dist_layes = dict(enumerate(nx.bfs_layers(G, [v])))
            bsf_tree = nx.reverse(nx.bfs_tree(G, source=v, depth_limit=None, sort_neighbors=random_neighbours))
            
            
            for i in range(self.k2):
                d = int(2*math.pow(3, i)*self.k1)
                        
                if not d in dist_layes.keys():
                    #no nodes in d-th level set
                    continue

                for u in dist_layes[d]:
                    distance_graphs[i].add_edge(v, u)

                    edges_on_u_v_path = list(nx.dfs_edges(bsf_tree, source=u))
                    
                    edge_feature_id = []
                    for e in edges_on_u_v_path:
                        edge_feature_id.append(edge_ids[e])
                    distance_graphs_attr_ids[i].append(edge_feature_id)
        
        data = DilatedInfoData(data.x, data.edge_index, data.edge_attr, data.y, data.pos)

        for i in range(self.k2):
            data.__setattr__(f'dilated_step_{i}_edge_index',  from_networkx(distance_graphs[i]).edge_index)

            idx = torch.tensor(distance_graphs_attr_ids[i], dtype=torch.long)

            if idx.shape[0] == 0: 
               # When no path is present, to prevent collate_fn issue, we need to create and empty feature vector id same size
               # [0, path_len]
               d = int(2*math.pow(3, i)*self.k1)
               empty_ids = torch.empty((0, d), dtype=torch.long)
               data.__setattr__(f'dilated_step_{i}_path_ids', empty_ids)
            else:
                data.__setattr__(f'dilated_step_{i}_path_ids',  idx)

        return data


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k1={self.k1}, k2={self.k2})')