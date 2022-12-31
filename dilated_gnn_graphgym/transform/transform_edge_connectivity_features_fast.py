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

import graph_tool as gt
from graph_tool.search import BFSVisitor, bfs_search, dfs_iterator

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


import numpy as np

class TreeEdgesRecorder(BFSVisitor):

    def __init__(self, num_vertices, g):
        self.g = g
        self.distances = np.full((num_vertices,), fill_value=0)
        self.node_at_dist = {}
        self.bsf_tree_reversed = gt.Graph()
        self.bsf_tree_reversed.add_vertex(num_vertices)

    def tree_edge(self, edge):
        e0 = self.g.vertex_index[edge.source()]
        e1 =  self.g.vertex_index[edge.target()]
        d = self.distances[e0] + 1
        self.distances[e1] = d
        if not d in self.node_at_dist:
            self.node_at_dist[d] = [e1]
        else:
            self.node_at_dist[d].append(e1)
        
        v1 = self.bsf_tree_reversed.vertex(e1)
        v0 = self.bsf_tree_reversed.vertex(e0)
        self.bsf_tree_reversed.add_edge(v1, v0)

        



class EdgeConnectivityAndFeaturesFast(BaseTransform):
    def __init__(self, k1, k2) -> None:
        super().__init__()
        self.k1 = k1 
        self.k2 = k2

    def __call__(self, data: Data) -> Data:
        
        if data.x is None:
            data.x = torch.zeros((data.num_nodes,1), dtype=torch.long)
            
        def random_neighbours(l):
            a = list(l)
            random.shuffle(a)
            return iter(a)


        if data.is_directed():
            raise ValueError("Graph is directed, cannot proceed!")
        
        #G = to_networkx(data, to_undirected=True, remove_self_loops=True)
        
        G = gt.Graph()
        G.add_vertex(data.num_nodes)
        edges = data.edge_index.t().tolist()
        G.add_edge_list(edges)
       

        distance_graphs = []
        distance_graphs_attr_ids = []
        for i in range(self.k2):
            distance_graphs.append(nx.Graph())
            distance_graphs_attr_ids.append([])

        # edge_ids = {}
        # for i in range(data.num_edges):
        #     e = (edges[i][0], edges[i][1])
        #     edge_ids[e] = i #TODO check here

        edges = [] #save memory


        for v in range(data.num_nodes):
            vis = TreeEdgesRecorder(data.num_nodes, G)
            tmp = bfs_search(G, G.vertex(v), vis)

            dist_layes = vis.node_at_dist
            bsf_tree = vis.bsf_tree_reversed

            for i in range(self.k2):
                d = int(2*math.pow(3, i)*self.k1)
                        
                if not d in dist_layes.keys():
                    #no nodes in d-th level set
                    continue

                for u in dist_layes[d]:
                    distance_graphs[i].add_edge(v, u)

                    edges_on_u_v_path = [(e.source(), e.target()) for e in dfs_iterator(G, G.vertex(u))]
                
                    edge_feature_id = []
                    for e in edges_on_u_v_path:
                        pass
                        #edge_feature_id.append(edge_ids[e])
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