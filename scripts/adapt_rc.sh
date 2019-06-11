export DATA_PATH=/home/jonatas/data/
# export DATA_PATH=/opt/jonatas/datasets/lavse/
export OUT_PATH=/opt/jonatas/runs/
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2

# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/f30k/adapt_i2t/beta_0.995/resnet101rc_gru/ \
# --sim adaptive_i2t_bn_linear \
# --loader image \
# --workers 4 \
# --image_encoder resnet101 \
# --text_encoder gru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.995 \
# --vocab vocab/f30k.json \
# --valid_interval 250 \
# --ngpu $NGPUS \
# --eval_before_training \
# --nb_epochs 30 \
# --early_stop 30 \
# --lr_decay_interval 15 \


# python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 9991 \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/f30k/adapt_i2t/beta_0.995/resnet50rc_gru/ \
# --sim adaptive_i2t_bn_linear \
# --loader image \
# --workers 4 \
# --image_encoder resnet50 \
# --text_encoder gru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.995 \
# --vocab vocab/f30k.json \
# --valid_interval 250 \
# --ngpu $NGPUS \
# --eval_before_training \
# --nb_epochs 30 \
# --early_stop 30 \
# --lr_decay_interval 15 \


# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/f30k/adapt_i2t/attngru_resnet152_rc/ \
# --sim adaptive_i2t_bn_linear \
# --loader image \
# --workers 4 \
# --image_encoder resnet152 \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.995 \
# --vocab vocab/f30k.json \
# --valid_interval 250 \
# --ngpu $NGPUS \
# --eval_before_training \
# --nb_epochs 30 \
# --early_stop 30 \
# --lr_decay_interval 15 \



python -m torch.distributed.launch --nproc_per_node=$NGPUS \
train.py \
--data_path $DATA_PATH \
--train_data coco.en \
--val_data coco.en \
--outpath $OUT_PATH/coco/adapt_i2t/ab_resnet152rc_gru/ \
--sim adaptive_i2t_bn_linear \
--loader image \
--workers 2 \
--image_encoder resnet152 \
--text_encoder gru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.995 \
--vocab vocab/coco.json \
--valid_interval 250 \
--ngpu $NGPUS \
--nb_epochs 45 \
--early_stop 30 \
--lr_decay_interval 15 \
# --eval_before_training \



# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/try/resnet152_attngru/ \
# --sim cosine \
# --loader image \
# --workers 8 \
# --image_encoder resnet152 \
# --text_encoder attngru \
# --text_pooling mean \
# --image_pooling mean \
# --lr 6e-4 \
# --beta .992 \
# --vocab vocab/f30k.json \
# --valid_interval 500 \
# --ngpu $NGPUS \
# --nb_epochs 30 \
# --early_stop 30 \
# --lr_decay_interval 15 \
# # --eval_before_training \
