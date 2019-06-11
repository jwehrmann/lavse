# export DATA_PATH=/home/jonatas/data/
# export OUT_PATH=/opt/jonatas/runs/
# export CUDA_VISIBLE_DEVICES=0,1
# export NGPUS=2

# python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 9991 \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/f30k/adapt_i2t_g4/resnet152rc_gru/ \
# --sim adapt_i2t_g4 \
# --loader image \
# --workers 8 \
# --image_encoder resnet152 \
# --text_encoder gru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-5 \
# --beta 0.995 \
# --vocab vocab/f30k.json \
# --valid_interval 250 \
# --ngpu $NGPUS \
# --early_stop 30 \
# --nb_epochs 45 \
# --lr_decay_interval 15 \

# export DATA_PATH=/opt/jonatas/datasets/lavse/
# export OUT_PATH=/opt/jonatas/runs/
# export CUDA_VISIBLE_DEVICES=2,3
# export NGPUS=2

# python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 9992 \
# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath $OUT_PATH/f30k_precomp/adapt_i2t_g4/hierarchical/ \
# --sim cosine \
# --loader precomp \
# --workers 1 \
# --image_encoder hierarchical \
# --text_encoder gru \
# --text_pooling mean \
# --image_pooling mean \
# --lr 6e-5 \
# --beta 0.995 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 250 \
# --ngpu 1 \
# --early_stop 30 \
# --nb_epochs 45 \
# --lr_decay_interval 15

# python train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --resume $OUT_PATH/f30k/adapt_i2t/beta_0.995/resnet101rc_gru/checkpoint_-1.pkl \
# --outpath $OUT_PATH/f30k/adapt_i2t/beta_0.995/resnet101ft_gru/ \
# --sim adaptive_i2t_bn_linear \
# --loader image \
# --workers 12 \
# --image_encoder resnet101_ft \
# --text_encoder gru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-5 \
# --beta 0.5 \
# --vocab vocab/f30k.json \
# --valid_interval 500 \
# --ngpu 1 \
# --nb_epochs 30 \
# --early_stop 30 \
# --lr_decay_interval 15 \
# --finetune \
# --data_parallel

export DATA_PATH=/home/jonatas/data/
# export DATA_PATH=/opt/jonatas/datasets/lavse/
export OUT_PATH=/opt/jonatas/runs/
export CUDA_VISIBLE_DEVICES=2,3
export NGPUS=2

python -m torch.distributed.launch --nproc_per_node=$NGPUS  --master_port 8921 \
train.py \
--data_path $DATA_PATH \
--train_data f30k.en \
--val_data f30k.en \
--outpath $OUT_PATH/f30k_rerun/adapt_i2t/ab_resnet152rc_gru/ \
--sim adapt_i2t \
--loader image \
--workers 2 \
--image_encoder resnet152 \
--text_encoder gru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.995 \
--vocab vocab/f30k.json \
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

