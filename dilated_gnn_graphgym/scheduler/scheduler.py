from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional

from torch.nn import Parameter
from torch.optim import SGD, Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR
from torch_geometric.graphgym.config import cfg
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import from_config
from yacs.config import CfgNode as CN

cfg.optim.step_size = 1
cfg.optim.step_gamma = 1.0
@register.register_scheduler('step_lr_epochs')
def none_scheduler(optimizer: Optimizer, step_size: int, step_gamma:int) -> StepLR:
    return StepLR(optimizer, step_size=step_size, gamma=step_gamma)