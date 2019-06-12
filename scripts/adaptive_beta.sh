
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/runs/ablation/beta_0/f30k_precomp.en/ \
--sim adaptive_i2t \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.0 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
# --eval_before_training


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/runs/ablation/beta_0.9/f30k_precomp.en/ \
--sim adaptive_i2t \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.9 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
# --eval_before_training


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/runs/ablation/beta_0.99/f30k_precomp.en/ \
--sim adaptive_i2t \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.99 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
# --eval_before_training


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/runs/ablation/beta_0.999/f30k_precomp.en/ \
--sim adaptive_i2t \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
# --eval_before_training




python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/runs/ablation/beta_0.9999/f30k_precomp.en/ \
--sim adaptive_i2t \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.9999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
# --eval_before_training
