CONFIG_DIR=$1
REPEAT=$2


for CONFIG in "$CONFIG_DIR"/*.yaml; do
  if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
      echo $CONFIG
      python agg_run.py --cfg $CONFIG 
  fi
done