export DATA_PATH=/opt/jonatas/datasets/lavse/
export OUT_PATH=/opt/jonatas/runs/
export CUDA_VISIBLE_DEVICES=1,2
export NGPUS=2

python -m torch.distributed.launch --nproc_per_node=$NGPUS \
train.py \
--data_path $DATA_PATH \
--train_data f30k.en \
--val_data f30k.en \
--outpath $OUT_PATH/f30k/vsepp_pt/ \
--profile vsepp \
--loader image \
--workers 10 \
--vocab vocab/f30k.json \
--valid_interval 250 \
--ngpu $NGPUS \
--nb_epochs 30
# --eval_before_training \
