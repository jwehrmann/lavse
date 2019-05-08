export DATA_PATH=/home/jonatas/data/lavse/
export OUT_PATH=runs/temp/


# # pythlocal

# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath $OUT_PATH/cross/lr_5e-4/f30k_precomp.en/ \
# --sim cross \
# --val_step 100 \
# --workers 0 \
# --image_encoder hierarchical \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 5e-4 \
# --beta 0.999 \
# --eval_before_training \


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath runs/adapt_conv_proj/f30k_precomp.en/ \
--workers 3 \
--sim adapt_conv_proj \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
--device cpu \
--loader dummy

# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath runs/adaptive_i2t_im_sa/f30k_precomp.en/ \
# --workers 3 \
# --sim rnn_proj_large \
# --image_encoder hierarchical \
# --text_encoder emb_proj \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.999 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \
# --device cpu \
# --loader dummy



# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath runs/adaptive_i2t_im_sa/f30k_precomp.en/ \
# --workers 3 \
# --sim rnn_proj \
# --image_encoder hierarchical \
# --text_encoder emb_proj \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.999 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \
# --device cpu \
# --loader dummy


# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath temp/adaptive_i2t/f30k_precomp.en/ \
# --sim rnn_proj \
# --workers 0 \
# --image_encoder hierarchical \
# --text_encoder embed_proj \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.999 \
# --vocab vocab/f30k_vocab.json \
# --batch_size 32 \
# --valid_interval 500 \
# # --eval_before_training


# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath runs/adaptive_test/f30k_precomp.en/ \
# --sim adaptive \
# --workers 3 \
# --image_encoder hierarchical \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --lr_decay_interval 5 \
# --lr_decay_rate 0.9 \
# --beta 0.999 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \
# --batch_size 128 \
# --eval_before_training


# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath runs/adaptive_vocab/lr2e-4/f30k_precomp.en/ \
# --sim adaptive \
# --workers 0 \
# --image_encoder hierarchical \
# --text_encoder convgru_sa \
# --text_pooling none \
# --image_pooling none \
# --lr 2e-4 \
# --beta 0.999 \
# --eval_before_training \
# --vocab vocab/f30k_vocab.json


# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath runs/cross/f30k_precomp.en/ \
# --sim cross \
# --workers 0 \
# --image_encoder hierarchical \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 2e-4 \
# --beta 0.999 \
# --eval_before_training \



# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath $OUT_PATH/temp_cosine/f30k_precomp.en/ \
# --sim cosine \
# --valid_interval 500 \
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
