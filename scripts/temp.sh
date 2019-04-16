export DATA_PATH=/home/jonatas/data/lavse/
export OUT_PATH=runs/temp/


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/adaptive/lr_2e-4/f30k_precomp.en/ \
--sim adaptive \
--val_step 100 \
--workers 0 \
--image_encoder scan \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 2e-4 \
--beta 0.999 \
--eval_before_training \


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
