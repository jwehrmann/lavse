export DATA_PATH=/home/jonatas/data/lavse/
export OUT_PATH=runs/temp/


# pythlocal

python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/cross/lr_5e-4/f30k_precomp.en/ \
--sim cross \
--val_step 100 \
--workers 0 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 5e-4 \
--beta 0.999 \
--eval_before_training \



# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath $OUT_PATH/sa/f30k_precomp.en/ \
# --sim cosine \
# --val_step 100 \
# --workers 0 \
# --image_encoder hierarchical \
# --text_encoder sa \
# --text_pooling mean \
# --image_pooling mean \
# --lr 5e-4 \
# --beta 0.999 \
# --eval_before_training \



# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath $OUT_PATH/img_sa_convgru/f30k_precomp.en/ \
# --sim cosine \
# --val_step 100 \
# --workers 0 \
# --image_encoder sa \
# --text_encoder convgru_sa \
# --text_pooling mean \
# --image_pooling mean \
# --lr_decay_interval 10 \
# --lr_decay_rate 0.1 \
# --lr 6e-4 \
# --beta 0.997 \
# --eval_before_training \


# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath $OUT_PATH/sagru_sa_convgru_norm_leaky_999/f30k_precomp.en/ \
# --sim cosine \
# --val_step 100 \
# --workers 0 \
# --image_encoder sagru \
# --text_encoder convgru_sa \
# --text_pooling mean \
# --image_pooling mean \
# --lr 2e-4 \
# --beta 0.999 \
# --eval_before_training \



# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath $OUT_PATH/sagru_sa_convgru_large_lr/f30k_precomp.en/ \
# --sim cosine \
# --val_step 100 \
# --workers 0 \
# --image_encoder sagru \
# --text_encoder convgru_sa \
# --text_pooling mean \
# --image_pooling mean \
# --lr_decay_interval 3 \
# --lr_decay_rate 0.9 \
# --lr 1e-3 \
# --beta 0.997 \
# --eval_before_training \

# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath $OUT_PATH/squeeze-hier/lr2e-4/f30k_precomp.en/ \
# --sim squeeze \
# --val_step 100 \
# --workers 0 \
# --image_encoder hierarchical \
# --text_pooling none \
# --image_pooling none \
# --lr 5e-4 \
# --beta 0.9995 \
# # --eval_before_training \
