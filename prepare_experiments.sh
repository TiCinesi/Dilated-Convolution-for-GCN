NAME=$1
GRID=$2
REPEAT=${3:-1}

python configs_gen.py --config configs_experiments/${NAME}.yaml \
  --grid configs_experiments/${GRID}.txt \
  --out_dir configs_experiments/

#bash parallel_euler.sh configs_experiments/${NAME}_grid_${GRID} $REPEAT