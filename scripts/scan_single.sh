export DATA_PATH=/home/jonatas/data/lavse/
export OUT_PATH=runs/temp/


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--profile scan \
--sim scan_i2t \
--outpath runs/baselin/scan_i2t_vocab/f30k_precomp.en/ \
--valid_interval 500 \
--workers 3 \
--vocab vocab/f30k_vocab.json \
# --eval_before_training \



python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.jt \
--val_data jap_precomp.jt \
--profile scan \
--sim scan_i2t \
--outpath $OUT_PATH/lavse/scan_i2t/jap_precomp.jt/ \
--valid_interval 500 \
--workers 0 \
--vocab vocab/complete.json \
