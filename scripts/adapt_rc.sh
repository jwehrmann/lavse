export DATA_PATH=/opt/jonatas/datasets/lavse/
export OUT_PATH=/opt/jonatas/runs/
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NGPUS=3


# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/f30k/adapt_i2t/beta_0.992_decay_10_160val/resnet50_rc/ \
# --sim adaptive_i2t_bn_linear \
# --loader image \
# --workers 4 \
# --image_encoder resnet50 \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.992 \
# --vocab vocab/f30k.json \
# --valid_interval 160 \
# --ngpu $NGPUS \
# --eval_before_training \
# --nb_epochs 30 \
# --early_stop 30 \
# --lr_decay_interval 10 \


# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/f30k/adapt_i2t/resnet50_rc/ \
# --sim adaptive_i2t_bn_linear \
# --loader image \
# --workers 4 \
# --image_encoder resnet50 \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.995 \
# --vocab vocab/f30k.json \
# --valid_interval 500 \
# --ngpu $NGPUS \
# --eval_before_training \
# --nb_epochs 30 \
# --early_stop 30 \
# --lr_decay_interval 15 \


python train.py \
--data_path $DATA_PATH \
--train_data f30k.en \
--val_data f30k.en \
--resume $OUT_PATH/f30k/adapt_i2t/resnet152_rc/checkpoint_-1.pkl \
--outpath $OUT_PATH/f30k/adapt_i2t/resnet152_ft/3e-5/cont/ \
--sim adaptive_i2t_bn_linear \
--loader image \
--workers 8 \
--image_encoder resnet152_ft \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 3e-5 \
--beta 0. \
--max_violation \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
--ngpu 1 \
--nb_epochs 30 \
--early_stop 30 \
--lr_decay_interval 15 \
--finetune \
--data_parallel \
--batch_size 100 \
--eval_before_training

# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --resume $OUT_PATH/f30k/adapt_i2t/resnet152_ft/2e-5/checkpoint_-1.pkl \
# --outpath $OUT_PATH/f30k/adapt_i2t/resnet152_ft/2e-5-cont/ \
# --sim adaptive_i2t_bn_linear \
# --loader image \
# --workers 10 \
# --image_encoder resnet152_ft \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 2e-5 \
# --beta 0. \
# --max_violation \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \
# --ngpu 1 \
# --nb_epochs 30 \
# --early_stop 30 \
# --lr_decay_interval 15 \
# --finetune \
# --data_parallel \
# --batch_size 100 \
# --eval_before_training