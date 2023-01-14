CONFIG_DIR="./configs_experiments/bottleneck_base_grid_depths_layers_small/"


for CONFIG in "$CONFIG_DIR"*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
      python main.py --cfg $CONFIG --repeat 1 --mark_done 
  fi
done