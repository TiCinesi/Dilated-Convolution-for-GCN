
mkdir logs

REPEAT=10
CONFIG_DIR="./configs_experiments/tu_base_grid_gin_cuda/"

for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_gcn_cuda/"

for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_sage_cuda/"

for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_sage_cpu/"
for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_gin_cpu/"
for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_gcn_cpu/"
for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

###POS

CONFIG_DIR="./configs_experiments/tu_base_grid_gin_pos_cuda/"

for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_gcn_pos_cuda/"

for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_sage_pos_cuda/"

for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_sage_pos_cpu/"
for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_gin_pos_cpu/"
for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_gcn_pos_cpu/"
for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done


CONFIG_DIR="./configs_experiments/tu_base_grid_gat_cpu/"
for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_gat_pos_cpu/"
for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_gat_pos_cuda/"
for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --gres=gpumem:40g --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done

CONFIG_DIR="./configs_experiments/tu_base_grid_gat_cuda/"
for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --gres=gpumem:20g --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=8 --output=logs/%j.out
  fi
done