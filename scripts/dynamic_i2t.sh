

python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath runs/dynamic_rnn_large/f30k_precomp.en/ \
--workers 3 \
--sim rnn_proj_large \
--image_encoder hierarchical \
--text_encoder emb_proj \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
--device cpu \
--loader dummy

python train.py \
--data_path $DATA_PATH \
--outpath $OUT_PATH/temp/dynamic_i2t_rnn_2/f30k_precomp.en/ \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--workers 3 \
--sim rnn_proj \
--image_encoder hierarchical \
--text_encoder emb_proj \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500


# python train.py \
# --data_path $DATA_PATH \
# --outpath $OUT_PATH/temp/dynamic_i2t/conv_proj_sa_256_g2/f30k_precomp.en/ \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --workers 3 \
# --sim conv_proj_sa_256_g2 \
# --image_encoder hierarchical \
# --text_encoder emb_proj \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.99 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \
# --eval_before_training


# python train.py \
# --data_path $DATA_PATH \
# --outpath $OUT_PATH/temp/dynamic_i2t/f30k_precomp.en/ \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --workers 3 \
# --sim conv_proj_sa \
# --image_encoder hierarchical \
# --text_encoder emb_proj \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.999 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \
# --eval_before_training

