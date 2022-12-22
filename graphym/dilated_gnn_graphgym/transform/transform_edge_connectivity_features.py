from torch_geometric.transforms import BaseTransform

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
from networkx import Graph
from networkx.algorithms.traversal.breadth_first_search import bfs_layers
import math 


class EdgeConnectivityAndFeatures(BaseTransform):
    def __init__(self, k1, k2) -> None:
        super().__init__()
        self.k1 = k1 
        self.k2 = k2

    def __call__(self, data: Data) -> Data:

        #TODO compute feature data (new class)
        #
        # print(sorted(list(nx.bfs_tree(H, source=6, depth_limit=None, sort_neighbors=random_neighbours).edges())))

        # Compute bsf tree
        # Find layer with nodes (distance = d)
        # Reverse edges, run DFS from each node
        # Get edge id via hash map
        # Create edge feature matrix with shape [num_edges, d, feature_size]
        #   (Or also can aggregate here, when p as hypeparameter)
        #


        if data.is_directed():
            raise ValueError("Graph is directed, cannot proceed!")
        
        graph = to_networkx(data, to_undirected=True, remove_self_loops=True)

        distance_graphs = []
        for _ in range(self.k2):
            distance_graphs.append(Graph())
        
        for node in range(data.num_nodes):
            dist_layes = dict(enumerate(bfs_layers(graph, [node])))

            for i in range(self.k2):
                d = 2*math.pow(3, i)*self.k1
                
                if not d in dist_layes.keys(): #new
                    continue

                for w in dist_layes[d]:
                    distance_graphs[i].add_edge(node, w)
        
    
        for i in range(self.k2):
            data.__setattr__(f'distance_graphs_{i}_edge_index',  from_networkx(distance_graphs[i]).edge_index)


        return data


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k1={self.k1}, '
                f'k2={self.k2})')