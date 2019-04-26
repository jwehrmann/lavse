
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath runs/ablation/adaptive_i2t_in/f30k_precomp.en/ \
--sim adaptive_i2t_in \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
--eval_before_training


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath runs/ablation/adaptive_i2t_bn_linear/f30k_precomp.en/ \
--sim adaptive_i2t_bn_linear \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
--eval_before_training


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath runs/ablation/adaptive_i2t_no_norm/f30k_precomp.en/ \
--sim adaptive_i2t_no_norm \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
--eval_before_training


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath runs/ablation/adaptive_i2t_no_norm_linear/f30k_precomp.en/ \
--sim adaptive_i2t_no_norm_linear \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
--eval_before_training
