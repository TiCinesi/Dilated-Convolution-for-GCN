CONFIG_DIR=$1

for EXP in "$CONFIG_DIR"/*; do
    echo $EXP
    python ./dilated_gnn_graphgym/agg_batch.py --dir $EXP --metric 'accuracy'
done