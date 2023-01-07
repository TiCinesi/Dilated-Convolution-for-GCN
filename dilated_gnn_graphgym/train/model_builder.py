import time
from typing import Any, Dict, Tuple

import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import LightningModule
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import network_dict, register_network


class GraphGymModule(LightningModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = network_dict[cfg.model.type](dim_in=dim_in,
                                                  dim_out=dim_out)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:

        if self.cfg.optim.scheduler == 'reduce_lr_on_plateau':
            optimizer = create_optimizer(self.model.parameters(), self.cfg.optim)
            scheduler = create_scheduler(optimizer, self.cfg.optim)
            if self.cfg.train.monitor_val:
                metric = f'val_{self.cfg.metric_best}'
            else:
                metric = self.cfg.metric_best
            return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": metric,
            },
        },
        else:
            optimizer = create_optimizer(self.model.parameters(), self.cfg.optim)
            scheduler = create_scheduler(optimizer, self.cfg.optim)
            return [optimizer], [scheduler]

    def _shared_step(self, batch, split: str) -> Dict:
        batch.split = split
        pred, true = self(batch)
        loss, pred_score = compute_loss(pred, true)
        step_end_time = time.time()
        return dict(loss=loss, true=true, pred_score=pred_score.detach(),
                    step_end_time=step_end_time)

    def training_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="train")

    def validation_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="val")

    def test_step(self, batch, *args, **kwargs):
        return self._shared_step(batch, split="test")

    @property
    def encoder(self) -> torch.nn.Module:
        return self.model.encoder

    @property
    def mp(self) -> torch.nn.Module:
        return self.model.mp

    @property
    def post_mp(self) -> torch.nn.Module:
        return self.model.post_mp

    @property
    def pre_mp(self) -> torch.nn.Module:
        return self.model.pre_mp


def create_model(to_device=True, dim_in=None, dim_out=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    Args:
        to_device (string): The devide that the model will be transferred to
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' in cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    if cfg.model.type == 'dilapos_gnn':
        if cfg.gnn.use_edge_features:
            cfg.gnn.layer_type_dilated = cfg.gnn.layer_type
        else:
            if cfg.dataset.positional_encoding_path:
                #Change to layer that support edges
                d = {'ginconv_paper':'edge_ginconv', 
                      'gcnconv':'edge_gcnconv',
                      'sageconv':'edge_sageconv',
                      'gatconv_paper':'edge_gatconv'
                    }
                cfg.gnn.layer_type_dilated= d[cfg.gnn.layer_type]
            else:
                cfg.gnn.layer_type_dilated = cfg.gnn.layer_type

    model = GraphGymModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model
