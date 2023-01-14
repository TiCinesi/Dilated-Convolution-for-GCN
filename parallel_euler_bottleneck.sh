CONFIG_DIR="./configs_experiments/bottleneck_base_grid_depths_layers/"


for CONFIG in "$CONFIG_DIR"bottleneck_base-name=depth_6-*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --gres=gpumem:20g --wrap="python main.py --cfg $CONFIG --repeat 1 --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=3GB --cpus-per-task=16 -A s_stud_infk --output=logs/%j.out
  fi
done


for CONFIG in "$CONFIG_DIR"bottleneck_base-name=depth_7-*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --gres=gpumem:30g --wrap="python main.py --cfg $CONFIG --repeat 1 --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=16GB --cpus-per-task=16 -A s_stud_infk --output=logs/%j.out
  fi
done