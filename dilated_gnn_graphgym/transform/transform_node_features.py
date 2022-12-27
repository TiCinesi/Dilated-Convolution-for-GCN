
import torch
from torch_geometric.transforms import BaseTransform

from torch_geometric.data import Data


class EmptyNodeFeatures(BaseTransform):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data: Data) -> Data:
        if data.x is None:
            data.x = torch.zeros((data.num_nodes,1), dtype=torch.float)
        return data