export DATA_PATH=/opt/jonatas/datasets/lavse/
export OUT_PATH=/opt/jonatas/runs/
export CUDA_VISIBLE_DEVICES=0,1,2
export NGPUS=3

# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# train.py \
# --data_path $DATA_PATH \
# --train_data birds.en \
# --val_data birds.en \
# --outpath $OUT_PATH/birds/adapt_i2t \
# --workers 1 \
# --sim adaptive_i2t_bn_linear \
# --loader birds \
# --image_encoder resnet152 \
# --text_encoder convgru_sa \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.995 \
# --vocab vocab/f30k_birds.json \
# --valid_interval 150 \
# --ngpu $NGPUS \
# --lr_decay_interval 10 \
# --batch_size 128


# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# train.py \
# --data_path $DATA_PATH \
# --train_data birds.en \
# --val_data birds.en \
# --outpath $OUT_PATH/birds/cosine \
# --workers 1 \
# --loader birds \
# --sim cosine \
# --image_encoder resnet152 \
# --text_encoder gru \
# --text_pooling lens \
# --image_pooling mean \
# --lr 6e-4 \
# --beta 0.995 \
# --vocab vocab/f30k_birds.json \
# --valid_interval 160 \
# --ngpu $NGPUS \
# --lr_decay_interval 10 \
# --batch_size 128

export CUDA_VISIBLE_DEVICES=3
export NGPUS=1
python train.py \
--data_path $DATA_PATH \
--train_data birds.en \
--val_data birds.en \
--outpath $OUT_PATH/birds/adapt_i2t_b64 \
--workers 2 \
--sim adaptive_i2t_bn_linear \
--loader birds \
--image_encoder resnet152 \
--text_encoder gru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.995 \
--vocab vocab/f30k_birds.json \
--valid_interval 300 \
--ngpu $NGPUS \
--lr_decay_interval 10 \
--batch_size 64 \
--eval_before_training
