mkdir logs
CONFIG_DIR="./configs_experiments/ogbg_molhiv_baseline_grid_gin/"

for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 --output=logs/%j.out
  fi
done