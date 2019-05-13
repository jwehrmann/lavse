
# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath runs/adaptive_t2i/f30k_precomp.en/ \
# --sim adaptive \
# --workers 2 \
# --image_encoder hierarchical \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.999 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \
# # --eval_before_training


# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath runs/adaptive_t2i_norm/f30k_precomp.en/ \
# --sim adaptive_norm \
# --workers 2 \
# --image_encoder hierarchical \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.999 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \
# --eval_before_training


# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath runs/adaptive_t2i_k4/f30k_precomp.en/ \
# --sim adaptive_k4 \
# --workers 2 \
# --image_encoder hierarchical \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.999 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \


# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath runs/adaptive_t2i_k4/f30k_precomp.en/ \
# --sim adaptive_k4 \
# --workers 2 \
# --image_encoder hierarchical \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.999 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \


# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath runs/temp/f30k_precomp.en/ \
# --sim adaptive \
# --workers 2 \
# --image_encoder hierarchical \
# --text_encoder attngru \
# --text_pooling mean \
# --image_pooling mean \
# --lr 6e-4 \
# --beta 0.999 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \
# --eval_before_training


python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en \
--outpath runs/adaptive_i2t_condvec_linear/coco_precomp.en/ \
--sim adaptive_i2t_condvec_linear \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru_cat \
--text_pooling mean \
--image_pooling mean \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/coco_vocab.json \
--valid_interval 500 \
--lr_decay_interval 7
# --eval_before_training


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/adaptive_i2t_bn_linear/attngru_cat/f30k_precomp.en/ \
--sim adaptive_i2t_bn_linear \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru_cat \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
--eval_before_training \
--save_all
attngru_cat_ek2


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/adaptive_i2t_bn_linear/attngru_cat_ek2/f30k_precomp.en/ \
--sim adaptive_i2t_bn_linear \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru_cat_ek2 \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
--eval_before_training \
--save_all


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/adaptive_i2t_bn_linear/attngru_cat_ek4/f30k_precomp.en/ \
--sim adaptive_i2t_bn_linear \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru_cat_ek4 \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
--eval_before_training \
--save_all


# Distributed
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/adaptive_i2t_bn_linear/f30k_precomp.en/ \
--sim adaptive_i2t_bn_linear \
--workers 3 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.99 \
--vocab vocab/f30k_vocab.json \
--valid_interval 250 \
--ngpu $NGPUS \
--eval_before_training \
