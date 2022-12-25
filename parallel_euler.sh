CONFIG_DIR=$1
REPEAT=$2


for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --wrap="python main.py --cfg $CONFIG --repeat $REPEAT" --time=12:00:00 --ntasks-per-node=1 --mem-per-cpu=1GB --cpus-per-task=16
  fi
done