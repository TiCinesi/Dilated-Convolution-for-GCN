import torch
from torch_geometric.transforms import BaseTransform

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import math
import numpy as np
from torch_geometric.graphgym.config import cfg

cfg.dataset.positional_encoding_path = False
class EdgeConnectivityPositional(BaseTransform):
    def __init__(self, k1, k2) -> None:
        super().__init__()
        self.k1 = k1 
        self.k2 = k2

    def __call__(self, data: Data) -> Data:
        if data.x is None:
            data.x = torch.zeros((data.num_nodes,1), dtype=torch.float)

        distance_graphs = []
        distance_graphs_dist = []
        distance_graphs_disconnected = []
        for i in range(self.k2):
            distance_graphs.append([[], []])
            distance_graphs_dist.append([])
            distance_graphs_disconnected.append([])
    

        if data.is_directed():
            raise ValueError("Not supported!")
        else:
            G = to_networkx(data, to_undirected=True, remove_self_loops=True)

            for v in range(G.number_of_nodes()):
                bsf_tree = nx.bfs_tree(G, source=v)
                dist_layes = dict(enumerate(nx.bfs_layers(bsf_tree, [v])))
                
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
                    
                    if  d in dist_layes.keys():
                        for u in dist_layes[d]:
                            distance_graphs[i][0].append(u)
                            distance_graphs[i][1].append(v)
                            distance_graphs_dist[i].append(d)
                            flag = False
                
                    distance_graphs_disconnected[i].append(flag)
        
        for i in range(self.k2):

            edges = torch.tensor(distance_graphs[i], dtype=torch.int64)
            d = torch.tensor(distance_graphs_dist[i],  dtype=torch.int64)

            if edges.shape[0] == 0: 
                edges = torch.empty((2,0), dtype=torch.int64)
            data.__setattr__(f'dilated_step_{i}_edge_index',  edges)

            if cfg.dataset.positional_encoding_path:
                data.__setattr__(f'dilated_step_{i}_path_len',  d)
                data.__setattr__(f'dilated_step_{i}_node_disconnected',  
                    torch.tensor(distance_graphs_disconnected[i], dtype=torch.bool))


        return data


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k1={self.k1}, '
                f'k2={self.k2})')