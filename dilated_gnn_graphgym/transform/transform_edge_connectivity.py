import torch
from torch_geometric.transforms import BaseTransform

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
import math

class EdgeConnectivity(BaseTransform):
    def __init__(self, k1, k2) -> None:
        super().__init__()
        self.k1 = k1 
        self.k2 = k2

    def __call__(self, data: Data) -> Data:

        #TODO compute feature data (new class)
        if data.is_directed():
            raise ValueError("Graph is directed, cannot proceed!")
        
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)

        distance_graphs = []
        for i in range(self.k2):
            distance_graphs.append(nx.Graph())
        
        for v in range(G.number_of_nodes()):
            dist_layes = dict(enumerate(nx.bfs_layers(G, [v])))

            for i in range(self.k2):
                d = int(2*math.pow(3, i)*self.k1)
                
                if not d in dist_layes.keys():
                    #no nodes in d-th level set
                    continue

                for u in dist_layes[d]:
                    distance_graphs[i].add_edge(v, u)
        
    
        for i in range(self.k2):
            data.__setattr__(f'dilated_step_{i}_edge_index',  from_networkx(distance_graphs[i]).edge_index)


        return data


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k1={self.k1}, '
                f'k2={self.k2})')