CONFIG_DIR=$1
REPEAT=$2


for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --gpus=1 --gres=gpumem:20g --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=24:00:00 --ntasks-per-node=1 --mem-per-cpu=2GB --cpus-per-task=16 -A s_stud_infk --output=logs/%j.out
  fi
done