CONFIG_DIR="./configs_experiments/bottleneck_base_grid_depths_layers/"

mkdir logs

for CONFIG in "$CONFIG_DIR"bottleneck_base-name=depth_2-*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat 1 --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 --output=logs/%j.out
  fi
done

for CONFIG in "$CONFIG_DIR"bottleneck_base-name=depth_3-*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat 1 --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 --output=logs/%j.out
  fi
done


for CONFIG in "$CONFIG_DIR"bottleneck_base-name=depth_4-*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat 1 --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 --output=logs/%j.out
  fi
done

for CONFIG in "$CONFIG_DIR"bottleneck_base-name=depth_5-*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat 1 --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 --output=logs/%j.out
  fi
done


for CONFIG in "$CONFIG_DIR"bottleneck_base-name=depth_6-*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --gres=gpumem:20g --wrap="python main.py --cfg $CONFIG --repeat 1 --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=16GB --cpus-per-task=16 --output=logs/%j.out
  fi
done


for CONFIG in "$CONFIG_DIR"bottleneck_base-name=depth_7-*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --gres=gpumem:30g --wrap="python main.py --cfg $CONFIG --repeat 1 --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=16GB --cpus-per-task=16 --output=logs/%j.out
  fi
done