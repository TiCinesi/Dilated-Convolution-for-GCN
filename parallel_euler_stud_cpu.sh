CONFIG_DIR=$1
REPEAT=$2


for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
    sbatch --wrap="python main.py --cfg $CONFIG --repeat $REPEAT --mark_done" --time=12:00:00 --ntasks-per-node=1 --mem-per-cpu=1GB --cpus-per-task=8 -A s_stud_infk
  fi
done