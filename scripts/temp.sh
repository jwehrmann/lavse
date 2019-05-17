export DATA_PATH=/opt/jonatas/datasets/lavse/
export OUT_PATH=/opt/jonatas/runs/
export CUDA_VISIBLE_DEVICES=1,2,3
export NGPUS=3

python -m torch.distributed.launch --nproc_per_node=$NGPUS \
train.py \
--data_path $DATA_PATH \
--train_data f30k.en \
--val_data f30k.en \
--outpath $OUT_PATH/temp/dyn_resnet152/ \
--workers 8 \
--loader image \
--sim dynconv \
--image_encoder resnet152 \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.995 \
--vocab vocab/f30k.json \
--valid_interval 160 \
--ngpu $NGPUS \
--lr_decay_interval 10 \
--batch_size 128

# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/temp/hier/resnet_101/f30k/ \
# --workers 8 \
# --loader image \
# --sim cosine \
# --image_encoder resnet101 \
# --text_encoder attngru \
# --text_pooling mean \
# --image_pooling mean \
# --lr 6e-4 \
# --beta 0.99 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 160 \
# --ngpu 1 \
# --eval_before_training


# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/f30k/adapt_resnet152/ \
# --sim adaptive_i2t_bn_linear \
# --loader image \
# --workers 4 \
# --image_encoder resnet152 \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.995 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \
# --ngpu $NGPUS \
# --eval_before_training \
# --nb_epochs 30 \
# --early_stop 30 \
# --lr_decay_interval 15 \


# export CUDA_VISIBLE_DEVICES=1,2,3
# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/temp/adapt_resnet152_ft/ \
# --sim adaptive_i2t_bn_linear \
# --loader image \
# --workers 10 \
# --image_encoder resnet152_ft \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 2e-5 \
# --beta 1. \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 500 \
# --ngpu 1 \
# --nb_epochs 30 \
# --early_stop 30 \
# --lr_decay_interval 15 \
# --finetune \
# --data_parallel \
# --batch_size 90
# --eval_before_training

# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/f30k/lr2e-5_max_violation/adapt-resume/ \
# --resume $OUT_PATH/f30k/adapt/checkpoint_0.pkl \
# --sim adaptive_i2t_bn_linear \
# --loader image \
# --workers 3 \
# --image_encoder resnet50 \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 2e-5 \
# --beta 1. \
# --max_violation \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 250 \
# --ngpu $NGPUS \
# --finetune \
# --batch_size 75 \
# --eval_before_training \

