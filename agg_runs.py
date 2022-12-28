import json
import logging
import os

import numpy as np

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.utils.io import (
    dict_list_to_json,
    dict_list_to_tb,
    dict_to_json,
    makedirs_rm_exist,
    string_to_python,
)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def is_seed(s):
    try:
        int(s)
        return True
    except Exception:
        return False


def is_split(s):
    if s in ['train', 'val', 'test']:
        return True
    else:
        return False


def join_list(l1, l2):
    assert len(l1) == len(l2), \
        'Results with different seeds must have the save format'
    for i in range(len(l1)):
        l1[i] += l2[i]
    return l1


def agg_dict_list(dict_list, best=False, epoch_name=None):
    """
    Aggregate a list of dictionaries: mean + std
    Args:
        dict_list: list of dictionaries

    """
    if best:
        dict_agg = {'epoch': 'val_best'}
    else:
        dict_agg = {'epoch': dict_list[0]['epoch']}

    if epoch_name is not None:
        dict_agg = {'epoch': epoch_name}

    for key in dict_list[0]:
        if key != 'epoch':
            value = np.array([dict[key] for dict in dict_list])
            dict_agg[key] = np.mean(value).round(cfg.round)
            dict_agg['{}_std'.format(key)] = np.std(value).round(cfg.round)
        else:
            if epoch_name is not None or best:
                value = np.array([dict[key] for dict in dict_list])
                dict_agg['ckpt_epoch'] = np.mean(value).round(cfg.round)
                dict_agg['{}_std'.format('ckpt_epoch')] = np.std(value).round(cfg.round)
        
    return dict_agg


def name_to_dict(run):
    run = run.split('-', 1)[-1]
    cols = run.split('=')
    keys, vals = [], []
    keys.append(cols[0])
    for col in cols[1:-1]:
        try:
            val, key = col.rsplit('-', 1)
        except Exception:
            print(col)
        keys.append(key)
        vals.append(string_to_python(val))
    vals.append(cols[-1])
    return dict(zip(keys, vals))


def rm_keys(dict, keys):
    for key in keys:
        dict.pop(key, None)


def json_to_dict_list(fname, split):
    dict_list = []
    #epoch_set = set()
    with open(fname) as f:
        lines = f.readlines()
        if split == 'val': #NOTE skip logs of sanity check dataloader
            lines = lines[1:]

        for line in lines:
            line = line.rstrip()
            dict = json.loads(line)
            # if dict['epoch'] not in epoch_set:
            dict_list.append(dict)
            #epoch_set.add(dict['epoch'])
    return dict_list

def agg_runs(dir, metric_best='auto'):
    r'''
    Aggregate over different random seeds of a single experiment

    Args:
        dir (str): Directory of the results, containing 1 experiment
        metric_best (str, optional): The metric for selecting the best
        validation performance. Options: auto, accuracy, auc.

    '''
    results = {'train': None, 'val': None, 'test': None}
    results_best = {'train': None, 'val': None, 'test': None}
    for seed in os.listdir(dir):
        if is_seed(seed):
            dir_seed = os.path.join(dir, seed)

            split = 'val'
            if split in os.listdir(dir_seed):
                dir_split = os.path.join(dir_seed, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                stats_list = json_to_dict_list(fname_stats, split)
                if metric_best == 'auto':
                    metric = 'auc' if 'auc' in stats_list[0] else 'accuracy'
                else:
                    metric = metric_best
                performance_np = np.array(  # noqa
                    [stats[metric] for stats in stats_list])
                best_epoch = \
                    stats_list[
                        eval("performance_np.{}()".format(cfg.metric_agg))][
                        'epoch']
                print(best_epoch)

            for split in os.listdir(dir_seed):
                if is_split(split):
                    dir_split = os.path.join(dir_seed, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    stats_list = json_to_dict_list(fname_stats, split)
                    
                    stats_best = [
                        stats for stats in stats_list
                        if stats['epoch'] == best_epoch
                    ][0]

                    print(stats_best)
                    stats_list = [[stats] for stats in stats_list]
                    if results[split] is None:
                        results[split] = stats_list
                    else:
                        results[split] = join_list(results[split], stats_list)
                    if results_best[split] is None:
                        results_best[split] = [stats_best]
                    else:
                        results_best[split] += [stats_best]
    results = {k: v for k, v in results.items() if v is not None}  # rm None
    results_best = {k: v
                    for k, v in results_best.items()
                    if v is not None}  # rm None
    for key in results:
        for i in range(len(results[key])):
            if key == 'test' and i == 0:
                results[key][i] = agg_dict_list(results[key][i], epoch_name='last_epoch')
            elif key == 'test' and i == 1:
                results[key][i] = agg_dict_list(results[key][i], epoch_name='best_epoch')
            else:
                results[key][i] = agg_dict_list(results[key][i])
    for key in results_best:
        results_best[key] = agg_dict_list(results_best[key], best=True)
    # save aggregated results
    for key, value in results.items():
        dir_out = os.path.join(dir, 'agg', key)
        makedirs_rm_exist(dir_out)
        fname = os.path.join(dir_out, 'stats.json')
        dict_list_to_json(value, fname)

        # if cfg.tensorboard_agg:
        #     if SummaryWriter is None:
        #         raise ImportError(
        #             'Tensorboard support requires `tensorboardX`.')
        #     writer = SummaryWriter(dir_out)
        #     dict_list_to_tb(value, writer)
        #     writer.close()
    for key, value in results_best.items():
        dir_out = os.path.join(dir, 'agg', key)
        fname = os.path.join(dir_out, 'best.json')
        dict_to_json(value, fname)
    logging.info('Results aggregated across runs saved in {}'.format(
        os.path.join(dir, 'agg')))

