GAT:
- 5649679



Depth 6:
- reults_experiments/bottleneck_base_grid_depths/bottleneck_base-name=depth_6-batch_size=1024-layers_k2=2/0        (dim=64)         dopo 75 epoche 0.1855
- results_experiments/bottleneck_base_grid_depths/bottleneck_base-name=depth_6-batch_size=1024-layers_k2=2-dim_inner=128    dopo 10mila 0.56
- results_experiments/bottleneck_base_grid_depths/bottleneck_base-name=depth_6-batch_size=1024-layers_k2=2-dim_inner=256 GPU 20G, RAM 32G    dopo 1 epoche 0.64 poi sceso terribile

Depth 7:
5649185 (depth7) 
configs_experiments/bottleneck_base_grid_depths/bottleneck_base-name=depth_7-batch_size=1024-layers_k2=3-dim_inner=128 GPU 20G RAM 16*16G (prob anche 10G GPU) 

results_experiments/bottleneck_base_grid_depths_2/bottleneck_base-name=depth_7-batch_size=512-accumulate_grad=4-layers_k2=2-dim_inner=128 dopo 70-10mila 0.5
results_experiments/bottleneck_base_grid_depths_2/bottleneck_base-name=depth_7-batch_size=1024-accumulate_grad=2-layers_k2=3-dim_inner=128 0.86

results_experiments/bottleneck_base_grid_depths_2/bottleneck_base-name=depth_7-batch_size=512-accumulate_grad=4-layers_k2=2-dim_inner=256/0                     0.94
results_experiments/bottleneck_base_grid_depths_2/bottleneck_base-name=depth_7-batch_size=1024-accumulate_grad=2-layers_k2=3-dim_inner=256/ 30G GPU,  16*16 RAM   1.0 

Depth 8:
- results_experiments/bottleneck_base_grid_depths/bottleneck_base-name=depth_8-batch_size=512-accumulate_grad=4-layers_k2=3 (64) dopo 2 epoche 0.024

5649562 (depth8_2)
- configs_experiments/bottleneck_base_grid_depths/bottleneck_base-name=depth_8-batch_size=512-accumulate_grad=4-layers_k2=3-dim_inner=128.yaml  GPU 30G RAM 8*32G 


  auto_resume: true
seed: 8


