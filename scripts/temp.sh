export DATA_PATH=/opt/jonatas/datasets/lavse/
export OUT_PATH=/opt/jonatas/runs/
export CUDA_VISIBLE_DEVICES=1,2,3
export NGPUS=3

python -m torch.distributed.launch --nproc_per_node=$NGPUS \
train.py \
--data_path $DATA_PATH \
--train_data f30k.en \
--val_data f30k.en \
--outpath $OUT_PATH/runs/adapt/resnet34_fixed/f30k/ \
--workers 4 \
--loader image \
--sim adaptive_i2t_bn_linear \
--image_encoder full_image \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.995 \
--vocab vocab/f30k_vocab.json \
--valid_interval 160 \
--ngpu $NGPUS \
--eval_before_training

# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k_precomp.en \
# --val_data f30k_precomp.en \
# --outpath $OUT_PATH/adaptive_i2t_bn_linear/f30k_precomp.en/ \
# --sim adaptive_i2t_bn_linear \
# --workers 3 \
# --image_encoder hierarchical \
# --text_encoder attngru \
# --text_pooling none \
# --image_pooling none \
# --lr 6e-4 \
# --beta 0.99 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 250 \
# --ngpu $NGPUS \
# --eval_before_training \


# python -m torch.distributed.launch --nproc_per_node=1 \
# train.py \
# --data_path $DATA_PATH \
# --train_data f30k.en \
# --val_data f30k.en \
# --outpath $OUT_PATH/temp/f30k/ \
# --workers 4 \
# --sim cosine \
# --image_encoder scan \
# --text_encoder scan \
# --text_pooling lens \
# --image_pooling mean \
# --lr 6e-4 \
# --beta 0.99 \
# --vocab vocab/f30k_vocab.json \
# --valid_interval 50 \
# --ngpu 1 \
