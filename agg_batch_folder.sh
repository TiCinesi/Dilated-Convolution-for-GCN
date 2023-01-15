CONFIG_DIR=$1

for EXP in "$CONFIG_DIR"/*; do
    echo $EXP
    python agg_batch.py --dir $EXP --metric 'accuracy'
done