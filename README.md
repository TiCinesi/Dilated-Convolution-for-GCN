# Unleashing the Potential of Dilated Convolution and Edge Features for GCN

Graph Convolutional Networks (GCN) are a popular approach for analyzing graph-structured data, but they can suffer from the issue of underreachability, where nodes that are far apart are unable to exchange information. We propose a new message passing framework for GCNs called dilated message passing, which overcomes the issue of underreachability by applying dilated convolutions to increase the receptive field of each node, allowing for information to exchange between nodes that were previously too far apart. Additionally, we develop a new stochastic method for creating edge features in the dilated convolution process. Our work provides new insights into the use of dilated convolution and advanced edge feature handling for analyzing graph data, and tries to address the problems of underreachability and oversquashing and significantly improves the performance of GCNNs on a variety of tasks.

#### Authors
Nathan Corecco, Giorgio Piatti, David Gu \
Department of Computer Science, ETH Zurich, Switzerland


## Overview

```
/configs                    -- Example configs for reference MP and our model
/configs_experiments        -- All configs files of the experiments that we have runned for our paper
/dilated_dnn_graphgym       -- Code with reference implementation (MP) and dilated message passing
/img                        -- Graphs for the paper
/notebooks                  -- Results analysis
/results_experiments        -- Aggregated results (for all logs regarding our experiment see below)
agg_batch_folder.sh
agg_run.py
main.py
prepare_experiments.sh
setup.sh
paper.pdf                   -- paper describing our approaches
```


#### Additional data
We already provided the aggregated score & hyperparameters of all our run in the results_experiment folder.
Extensive experiment logs (not aggregated) are available here : TODO-lin


### Installation
To run the code, it is recommended to first run the following instructions inside a conda enviroment:
`sh ./setup.sh`


## Development instructions
Our implementation is based on GraphGym [1] a framework to design and evaluate Grap Neural Networks. We extended the code quite a bit to make it more flexible, still the basic principle remain the same, for more details see https://github.com/snap-stanford/GraphGym .

We introduced some new components that we summarize next.
### New components

#### Layers


## In-depth Usage

### Run an experiment
**1.1 Specify a configuration file.**
In GraphGym, an experiment is fully specified by a `.yaml` file.
Unspecified configurations in the `.yaml` file will be populated by the default values in 
For example, in [`configs/example.yaml`](configs/example.yaml), 
there are configurations on dataset, training, model, GNN, etc.
Concrete description for each configuration that we added on top of GraphGym is described above.

**1.2 Launch an experiment.**
```bash
python main.py --cfg configs/example.yaml --repeat 10
```
You can specify the number of different random seeds to repeat via `--repeat`.

**1.3 Understand the results.**
Experimental results will be automatically saved in directory `results/${CONFIG_NAME}/`; 
in the example above, it is `results/example/`.
Results for different random seeds will be saved in different subdirectories, such as `results/example/2`.
The aggregated results over all the random seeds are *automatically* generated into `results/example/agg`, including the mean and standard deviation `_std` for each metric.
Train/val/test results are further saved into subdirectories, such as `results/example/agg/val`: 
- `stats.json` stores the results after each epoch aggregated across random seeds, 
- `best.json` stores the results at *the epoch with the highest validation metric*.

### Generate a batch of experiment configs
**2.1 Specify a base file.**
GraphGym supports running a batch of experiments. To start, a user needs to select a base architecture `--config`. The batch of experiments will be created by perturbing certain configurations of the base architecture.

**2.2 Specify a grid file.**
A grid file describes how to perturb the base file, in order to generate the batch of the experiments. For example, the base file could specify an experiment of 3-layer GCN for PROTEINS graph classification. Then, the grid file specifies how to perturb the experiment along different dimension, such as number of layers, model architecture, dataset, level of task, etc.

**2.4 Generate config files for the batch of experiments,** based on the information specified above.
```bash
python configs_gen.py --config configs_experiment/${DIR}/${CONFIG}.yaml \
  --grid grids/${DIR}/${GRID}.txt \
  --out_dir configs_experiment
```

**2.5 Launch the batch of experiments.**
In our experiment we used the Euler cluster, on the SLURM batch system. Helper files are already provided.
- Launch experiments which requires GPU
```bash
bash scripts_euler_helper/parallel_euler_stud.sh configs_experiment/${CONFIG}_grid_${GRID} $REPEAT
```
- Launch experiments which requires CPU only
```bash
bash scripts_euler_helper/parallel_euler_stud_cpu.sh configs_experiment/${CONFIG}_grid_${GRID} $REPEAT
```

## References
[1] You, J., Ying, Z., & Leskovec, J. (2020). Design space for graph neural networks. Advances in Neural Information Processing Systems, 33, 17009-17021.