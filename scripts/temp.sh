export DATA_PATH=/opt/jonatas/datasets/lavse/
export OUT_PATH=/opt/jonatas/runs/
export CUDA_VISIBLE_DEVICES=1,2,3
export NGPUS=3

# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/runs/dyn/resnet34_fixed/f30k/ \
# --workers 4 \
# --loader image \
# --sim dynconv \
# --image_encoder full_image \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 2e-4 \
# --beta 0.999 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 160 \
# --ngpu $NGPUS \
# --eval_before_training


python -m torch.distributed.launch --nproc_per_node=$NGPUS \
train.py \
--data_path $DATA_PATH \
--train_data f30k.en \
--val_data f30k.en \
--outpath $OUT_PATH/f30k/adapt/ \
--sim adaptive_i2t_bn_linear \
--loader image \
--workers 3 \
--image_encoder resnet50 \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.993 \
--vocab vocab/f30k_vocab.json \
--valid_interval 160 \
--ngpu $NGPUS \
--eval_before_training \


python -m torch.distributed.launch --nproc_per_node=$NGPUS \
train.py \
--data_path $DATA_PATH \
--train_data f30k.en \
--val_data f30k.en \
--outpath $OUT_PATH/f30k/scan/ \
--workers 4 \
--sim cosine \
--image_encoder resnet50 \
--text_encoder scan \
--text_pooling lens \
--image_pooling mean \
--lr 6e-4 \
--beta 0.993 \
--vocab vocab/f30k_vocab.json \
--valid_interval 160 \
--ngpu $NGPUS \
