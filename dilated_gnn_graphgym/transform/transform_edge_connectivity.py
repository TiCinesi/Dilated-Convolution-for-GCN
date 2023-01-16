import torch
from torch_geometric.transforms import BaseTransform

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import math

class DilatedMPGraphTransform(BaseTransform):
    def __init__(self, k1, k2) -> None:
        super().__init__()
        self.k1 = k1 
        self.k2 = k2

    def __call__(self, data: Data) -> Data:
        if data.x is None:
            data.x = torch.zeros((data.num_nodes,1), dtype=torch.float)

        distance_graphs = []
        for i in range(self.k2):
            distance_graphs.append([[], []])
    

        if data.is_directed():
            G = to_networkx(data, to_undirected=False, remove_self_loops=True)


            for v in range(G.number_of_nodes()):
                depth_limit = int(math.pow(2, self.k2-1)*self.k1)
                bsf_tree = nx.bfs_tree(nx.reverse(G), source=v, depth_limit=depth_limit)
                dist_layes = dict(enumerate(nx.bfs_layers(bsf_tree, [v])))
                bsf_tree = nx.reverse(bsf_tree)
                
                
                for i in range(self.k2):
                    d = int(math.pow(2, i)*self.k1)
                            
                    if not d in dist_layes.keys():
                        #no nodes in d-th level set
                        continue

                    for u in dist_layes[d]:
                        distance_graphs[i][0].append(u)
                        distance_graphs[i][1].append(v)
        
        else:
            G = to_networkx(data, to_undirected=True, remove_self_loops=True)

            for v in range(G.number_of_nodes()):
                depth_limit = int(2*math.pow(3, self.k2-1)*self.k1)
                bsf_tree = nx.bfs_tree(G, source=v, depth_limit=depth_limit)
                dist_layes = dict(enumerate(nx.bfs_layers(bsf_tree, [v])))
                
                for i in range(self.k2):
                    d = int(2*math.pow(3, i)*self.k1)
                            
                    if not d in dist_layes.keys():
                        #no nodes in d-th level set
                        continue

                    for u in dist_layes[d]:
                        distance_graphs[i][0].append(u)
                        distance_graphs[i][1].append(v)
        
        for i in range(self.k2):

            edges = torch.tensor(distance_graphs[i], dtype=torch.int64)
            if edges.shape[0] == 0: 
                edges = torch.empty((2,0), dtype=torch.int64)
            data.__setattr__(f'dilated_step_{i}_edge_index',  edges)


        return data


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k1={self.k1}, '
                f'k2={self.k2})')