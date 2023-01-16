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
import numpy as np
from torch_geometric.graphgym.config import cfg

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

cfg.dataset.positional_encoding_path  = False
class DilatedLeafsMPGraphEdgeFeaturesTransform(BaseTransform):
    def __init__(self, k1, k2) -> None:
        super().__init__()
        self.k1 = k1 
        self.k2 = k2

    def __call__(self, data: Data) -> Data:
        
        if data.x is None:
            data.x = torch.zeros((data.num_nodes,1), dtype=torch.float)
        def random_neighbours(l):
            a = list(l)
            random.shuffle(a)
            return iter(a)

        distance_graphs = []
        distance_graphs_attr_ids = []
        distance_graphs_dist = []
        distance_graphs_disconnected = []
        for i in range(self.k2):
            distance_graphs.append([[], []])
            distance_graphs_attr_ids.append([])
            distance_graphs_dist.append([])
            distance_graphs_disconnected.append([])

        edge_ids = {}
        #TODO speed up this part
        for i in range(data.edge_index.shape[1]):
            e = tuple(data.edge_index[:, i].numpy())
            edge_ids[e] = i 

        if data.is_directed():
            raise ValueError("Not supported!")
        else:
            G = to_networkx(data, to_undirected=True, remove_self_loops=True)

            for v in range(G.number_of_nodes()):
                bsf_tree = nx.bfs_tree(G, source=v, sort_neighbors=random_neighbours)
                dist_layes = dict(enumerate(nx.bfs_layers(bsf_tree, [v])))
                bsf_tree = nx.reverse(bsf_tree)
                
                dist_node = np.zeros((G.number_of_nodes(),))
                for (k,layer) in dist_layes.items():
                    for n in layer:
                        dist_node[n] = k
                
            
                for i in range(self.k2):
                    flag = True
                    d = int(2*math.pow(3, i)*self.k1)
                    d_know = int(math.pow(3, i)*self.k1)
                    
                    # find leaf < d & > d_know
                    for u, deg in bsf_tree.out_degree():
                        if deg == 0 and dist_node[u] > d_know and  dist_node[u] < d:
                            distance_graphs[i][0].append(u)
                            distance_graphs[i][1].append(v)
                            distance_graphs_dist[i].append(dist_node[u])
                            flag = False
                            edges_on_u_v_path = list(nx.dfs_edges(bsf_tree, source=u))
                        
                            edge_feature_id = []
                            for e in edges_on_u_v_path:
                                edge_feature_id.append(edge_ids[e])
                            distance_graphs_attr_ids[i].append(edge_feature_id)

                    if  d in dist_layes.keys():
                        for u in dist_layes[d]:
                            distance_graphs[i][0].append(u)
                            distance_graphs[i][1].append(v)
                            distance_graphs_dist[i].append(d)
                            flag = False
                            
                            edges_on_u_v_path = list(nx.dfs_edges(bsf_tree, source=u))
                            
                            edge_feature_id = []
                            for e in edges_on_u_v_path:
                                edge_feature_id.append(edge_ids[e])
                            distance_graphs_attr_ids[i].append(edge_feature_id)             
                    distance_graphs_disconnected[i].append(flag)
        
        data = DilatedInfoData(data.x, data.edge_index, data.edge_attr, data.y, data.pos)

        for i in range(self.k2):

            edges = torch.tensor(distance_graphs[i], dtype=torch.int64)
            d = torch.tensor(distance_graphs_dist[i],  dtype=torch.int64)
            
            if edges.shape[0] == 0: 
                edges = torch.empty((2,0), dtype=torch.int64)

            data.__setattr__(f'dilated_step_{i}_edge_index',  edges)

            if cfg.dataset.positional_encoding_path:
                data.__setattr__(f'dilated_step_{i}_path_len',  d)
                # data.__setattr__(f'dilated_step_{i}_node_disconnected',  
                #     torch.tensor(distance_graphs_disconnected[i], dtype=torch.bool))
            
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