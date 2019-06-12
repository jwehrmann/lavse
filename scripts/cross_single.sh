export DATA_PATH=/home/jonatas/data/lavse/
export OUT_PATH=runs/temp/


# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath $OUT_PATH/cross/f30k_precomp.en/ \
# --sim cross \
# --val_step 100 \
# --workers 4 \
# --image_encoder scan \
# --text_pooling none \
# --image_pooling none \
# --lr 2e-4 \
# --beta 0.995


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/x_attn/f30k_precomp.en/ \
--sim cross \
--val_step 100 \
--workers 4 \
--image_encoder scan \
--text_pooling none \
--image_pooling none \
--lr 2e-4 \
--beta 0.995