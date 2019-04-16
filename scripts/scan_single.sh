export DATA_PATH=/home/jonatas/data/lavse/
export OUT_PATH=runs/temp/


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--profile scan \
--sim scan_i2t \
--outpath $OUT_PATH/lavse/vsepp/f30k_precomp.en/