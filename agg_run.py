import logging
import os

import dilated_gnn_graphgym  # noqa, register custom modules
import torch

from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
    get_fname
)
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import GraphGymDataModule, train
from dilated_gnn_graphgym import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)

    # Aggregate results from different seeds
    fname = get_fname(args.cfg_file)
    cfg.out_dir = os.path.join(cfg.out_dir, fname)
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
