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


### Dependencies
This conde depends on the following libraries
```
pytorch                   1.13.1         
pytorch-cluster           1.6.0           
pytorch-cuda              11.7               
pytorch-lightning         1.8.6                               
pytorch-scatter           2.1.0     
pytorch-sparse            0.6.16 
yacs                      0.1.8
ogb                       1.3.5
networkx                  2.8.8
numpy                     1.23.4
pandas                    1.5.2 
plotly                    5.12.0

```

To run the code, it is recommended to install all dependencies in a conda enviroment, we provid a file to setup all libraries.
```
bash ./setup.sh
```

## Datasets
Datasets are either dowloaded by the code at runtime or generate at runtime (if syntetic). We use the TUDataset via the `torch_geometric` library and  `ogbg-molhiv` via the library `ogb`.
The dataset Tree-Neighboursmatch is generate on the fly using an adapted implementation from Alon et al. [2].
TODO dataset summars

## How to reproduce our experiment
Our experiments run on a mixture of requirements, depending on the complexity of the model and the size of the dataset. To reproduce our experiments, we provide scripts that queue them on a SLURM cluster, since all experiments were performed on the ETH Euler cluster.

- Comparison across dataset and layers: `bash ./run_experiments_layers_across_datasets.sh`
- ogbg-molhiv experiments:  `bash ./run_experiments_ogbg_molhiv.sh`
- Tree-Neighboursmatch experiments:  `bash ./run_experiments_bottleneck.sh`

## How our implementation works
Our implementation is based on GraphGym [1] a framework to design and evaluate Grap Neural Networks. We extended the code quite a bit to make it more flexible, still the basic principle remain the same, for more details see https://github.com/snap-stanford/GraphGym .

We introduced some new components that we summarize next.

#### Models
| `model.name` | Description                                             | Notes
| ------------ | ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| standard_gnn | Standard message passing model                          | Standard model that uses message passing                                                                                         |
| dilated_gnn  | Dilated model  (See section III of the paper)           | Uses data transformer `DilatedMPGraphEdgeFeaturesTransform` if `gnn.edges_features` is `true` else               `DilatedMPGraphTransform`                    |
| dilapos_gnn  | Dilated model with leafs (See section III of the paper) | Uses data transformer `DilatedLeafsMPGraphEdgeFeaturesTransform` if `gnn.edges_features` is `true` else `DilatedLeafsMPGraphTransform`|

Configuration options for standard and dilated model:
- `gnn.layers_pre_mp`: number of layers before message passing (default = 0)
- `gnn.layer_type`: layer used in message passing
- `gnn.layers_post_mp`: number of layers after message passing (default = 1)
- `gnn.dim_inner`: dimension of node embeddings during message passsing
- `gnn.batchnorm`: batchnormalization between layers (default = true)
- `gnn.act`: activation function (default = relu)
- `gnn.act_on_last_layer_mp`: use activation function on last layers of message passing (both standard and dilated) (defaull = false)
- `gnn.dropout`:  dropout p (default = 0.0)

Options for dilated model:
- `gnn.layers_k1`: number of layers for first phase, standard message passing
- `gnn.layers_k2`: number of layers for second phase, dilated message passing
- `gnn.dilated_path_join`: how to join standard phase and dilated phase, possible values `add` (default), `concat`
- `gnn.learn_alpha_residual_connection`: whether to learn alphas coefficient in residual connections in the dilated phase (default = true)

- `gnn.use_edge_features`: need to be set true when using dataset with edge features
- `gnn.edge_agg`: see layers section (only used when: gnn.use_edge_features=true )

- `dataset.positional_encoding_path`: use positional econding to encode the path length on the edge attributes, as described in the subsection "Strategies for enhancing performance" (Section III) in the paper. Available only when `model.name` is `dilapos_gnn`.

Configuration options for standard model:
- `gnn.layers_mp`: number of layers of standard message passing


#### Dataloader & Data transformer
Since our dilated messagge passing phase, at layer `l`,  is based on the `2k_1*3^(l-1)` neighbor set of v, we compute this information and store in the torch_geometric data object.
This tranformation is performed using the algorithm package `networkx`.


| Data transfomer                          | Description                                                                                                                                                  |
| ---------------------------------------  | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DilatedMPGraphTransform                  | For each l-th dialted layer, add a set of edges, which can be used to aggregate message as describe in Eq (2) of the paper.                                  |
| DilatedMPGraphEdgeFeaturesTransform      | As above, but also saves the edge features along the path. (see paper
| DilatedLeafsMPGraphTransform             | For each l-th dialted layer, add a set of edges, which can be used to aggregate message as describe in Eq (2) with modification given by Eq (5) of the paper.    Moreover it saves for each edge the lenght of the path. |
| DilatedLeafsMPGraphEdgeFeaturesTransform | As above, but also saves the edge features along the path. (see paper)                                                                                        |

#### Layers
|`gnn.layer_type`| Edge Features (i.e. dataset.use_edge_features: true ) | `gnn.edge_agg` options | Other options     | Reference |
| -------------  | ----- | ----------------- | ---------------------------------------------------------------------- | --------- |
| ginconv_paper  | False |                   |                                 -                                      | [6]       |
| edge_ginconv   | True  | only 'add'        |                                 -                                      | [6,7]     |
| gatconv_paper  | False |                   | # heads `gnn.att_heads`,  #heads for final layer `gnn.att_heads_final` | [3]       |
| edge_gatconv   | True  | Not used.         | # heads `gnn.att_heads`,  #heads for final layer `gnn.att_heads_final` | [3]       |
| gcnconv        | False |                   |                                 -                                      | [5]       |
| edge_gcnconv   | True  | 'add', 'concat'   |                                 -                                      | [5,7]     |
| sageconv       | False |                   |                                 -                                      | [4]       |
| edge_sageconv  | True  | 'add', 'concat'   |                                 -                                      | [4,7]     |

In the original paper of GIN, GraphSAGE and GCN there is no mention of doing messagge passing with edge features, we extented the layer following the approach used by Hu et al. in "Strategies for pre-training graph neural networks" [7].


#### Pooling
|     `model.graph_pooling`   |            Description         | Reference |
| --------------------------- | ------------------------------ | --------  |
| mean                        | Mean pooling of GraphGym       |  -        |
| max_of_concat_layers        | Concat across layers, then max | [8]       |
| concat_across_sum_of_layers | GIN-Pooling                    | [6]       |


#### Schedulers
|     `optim.scheduler`       |                          Description                         |
| --------------------------- | ------------------------------------------------------------ |
| step_lr_epochs              | Wrapper around `torch.optim.lr_scheduler.StepLR`             |
| reduce_lr_on_plateau        | Wrapper around `torch.optim.lr_scheduler.ReduceLROnPlateau`  |

#### Syntetic dataset (Tree-Neighboursmatch) specific components
Encoder for Tree-Neighboursmatch:  `bottleneck_encoder`
Head for Tree-Neighboursmatch: `bottleneck_head`, classification head for the root.
See [`configs/example_bottleneck.yaml`](configs/example_bottleneck.yaml) as example configuration on how to run experiment on the Tree-Neighboursmatch problem with our dilated messagge passing model.


## In-depth Usage Framework

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
[1] You, Jiaxuan, Zhitao Ying, and Jure Leskovec. "Design space for graph neural networks." Advances in Neural Information Processing Systems 33 (2020): 17009-17021.

[2] Alon, Uri, and Eran Yahav. "On the bottleneck of graph neural networks and its practical implications." arXiv preprint arXiv:2006.05205 (2020).

[3] Veličković, Petar, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. "Graph attention networks." arXiv preprint arXiv:1710.10903 (2017).

[4] Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." Advances in neural information processing systems 30 (2017).

[5] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016).

[6] Xu, Keyulu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. "How powerful are graph neural networks?." arXiv preprint arXiv:1810.00826 (2018).

[7] Hu, Weihua, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. "Strategies for pre-training graph neural networks." arXiv preprint arXiv:1905.12265 (2019).

[8] Zhang, Shuo, and Lei Xie. "Improving attention mechanism in graph neural networks via cardinality preservation." In IJCAI: Proceedings of the Conference, vol. 2020, p. 1395. NIH Public Access, 2020.