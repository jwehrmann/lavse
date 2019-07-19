
export CUDA_VISIBLE_DEVICES=3
export NGPUS=1

# Single dataset
# python -m torch.distributed.launch  --nproc_per_node=$NGPUS --master_port 9992 \
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/lavse/clmr_gru/f30k_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder gru \
--vocab_path vocab/complete.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr word \
--text_pooling lens \


python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en \
--outpath $OUT_PATH/lavse/clmr_gru/coco_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder gru \
--vocab_path vocab/complete.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr word \
--text_pooling lens \


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/lavse/img_gru/f30k_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder gru \
--vocab_path vocab/complete.json \
--early_stop 100 \
--image_encoder img_gru \
--text_repr word \
--text_pooling lens \



python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en \
--outpath $OUT_PATH/lavse/multihead_k8/coco_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 0 \
--text_encoder gru \
--vocab_path vocab/complete.json \
--early_stop 100 \
--image_encoder multihead \
--text_repr word \
--text_pooling lens \
--lr_decay_interval 7



python train.py \
--data_path $DATA_PATH \
--train_data m30k_precomp.de \
--val_data m30k_precomp.de \
--outpath $OUT_PATH/lavse/clmr_gru/m30k_precomp.de/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder gru \
--vocab_path vocab/complete.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr word \
--text_pooling lens \


export CUDA_VISIBLE_DEVICES=2


python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.en \
--val_data jap_precomp.en jap_precomp.jt \
--adapt_data jap_precomp.en-jt \
--outpath $OUT_PATH/lavse/liwe_gru_128_512/jap_precomp.en-jt/ \
--beta 0.995 \
--lr 6e-4 \
--workers 0 \
--text_encoder liwe_gru_128_512 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling lens \


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de \
--adapt_data m30k_precomp.en-de \
--outpath $OUT_PATH/lavse/liwe_gru_384_384/m30k_precomp.en-de/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder liwe_gru_384_384 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling lens \



python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de \
--adapt_data m30k_precomp.en-de \
--outpath $OUT_PATH/lavse/liwe_gru_gru_proj/m30k_precomp.en-de/ \
--beta 0.991 \
--lr 6e-4 \
--workers 0 \
--text_encoder liwe_gru_gru_proj \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling lens \



python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/runs/ablation/beta_0.99/f30k_precomp.en/ \
--sim adaptive_i2t \
--workers 2 \
--image_encoder hierarchical \
--text_encoder attngru \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.99 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500 \
# --eval_before_training


python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en \
--outpath $OUT_PATH/lavse/adapt_i2t_img_gru/coco_precomp.en/ \
--sim adaptive_i2t \
--beta 0.991 \
--lr 6e-4 \
--workers 0 \
--text_encoder gru \
--vocab_path vocab/coco_vocab.json \
--early_stop 100 \
--image_encoder img_gru \
--text_pooling none \
--image_pooling none \
--lr_decay_interval 7



python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en \
--outpath $OUT_PATH/lavse/liwe_gru_128_512_img_gru/coco_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 0 \
--text_encoder liwe_gru_128_512 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder img_gru \
--text_repr liwe \
--text_pooling lens \
--lr_decay_interval 7


python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en \
--outpath $OUT_PATH/lavse/liwe_gru_384_384/coco_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder liwe_gru_384_384 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling lens \


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/lavse/liwe_gru_384_384_gelu_b991/f30k_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 0 \
--text_encoder liwe_gru_384_384_gelu \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling lens \



python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/lavse/liwe_gru_384_384_gelu_withbn/f30k_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder liwe_gru_384_384_gelu_withbn \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling lens \



python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/iccv/lavse/liwe_gru_256_512/f30k_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder liwe_gru_256_512 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling lens \


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/lavse/liwe_gru_scale_384_relu/f30k_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 0 \
--text_encoder liwe_gru_scale_384_relu \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling lens \



python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/lavse/liwe_gru_scale_384_gelu_nobn/f30k_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 0 \
--text_encoder liwe_gru_scale_384_gelu_nobn \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling lens \




python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en \
--outpath $OUT_PATH/lavse/liwe_gru_scale_384/coco_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 0 \
--text_encoder liwe_gru_scale_384 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling lens \



python test.py --data_path $DATA_PATH --model_path $OUT_PATH --val_data coco_precomp.en --vocab_path vocab/char.json --text_repr liwe --outpath results/



python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/lavse/scan_i2t_liwe_384_384/f30k_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 0 \
--text_encoder liwe_gru_384 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder scan \
--text_repr liwe \
--text_pooling none \
--image_pooling none \
--sim scan_i2t \


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/lavse/liwe_convgru_384_384/f30k_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder liwe_convgru_384_384 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling mean \



python train.py \
--data_path $DATA_PATH \
--train_data m30k_precomp.de \
--val_data m30k_precomp.de \
--outpath $OUT_PATH/lavse/liwe_gru_384_384/m30k_precomp.de/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder liwe_gru_384_384 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling lens \


export CUDA_VISIBLE_DEVICES=0
export NGPUS=1

# Single dataset
# python -m torch.distributed.launch  --nproc_per_node=$NGPUS --master_port 9992 \
python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en \
--outpath $OUT_PATH/lavse/liwe_gru_384/coco_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder liwe_gru_384 \
--ngpu $NGPUS \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling mean \


export CUDA_VISIBLE_DEVICES=2
export NGPUS=1

# Single dataset
# python -m torch.distributed.launch  --nproc_per_node=$NGPUS --master_port 9992 \
python train.py \
--data_path $DATA_PATH \
--train_data m30k_precomp.de \
--val_data m30k_precomp.de \
--outpath $OUT_PATH/lavse/liwe_gru_384/m30k_precomp.de/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder liwe_gru_384 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling mean \



# Single dataset
# python -m torch.distributed.launch  --nproc_per_node=$NGPUS --master_port 9992 \
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/lavse/liwe_gru_256_max/f30k_precomp.en/ \
--beta 0.991 \
--max_violation \
--lr 6e-4 \
--workers 1 \
--text_encoder liwe_gru_384 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling mean \


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/lavse/liwe_gru_256_sum/f30k_precomp.en/ \
--beta 1. \
--lr 6e-4 \
--workers 1 \
--text_encoder liwe_gru_384 \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling mean \




export CUDA_VISIBLE_DEVICES=1
export NGPUS=1

# Single dataset
# python -m torch.distributed.launch  --nproc_per_node=$NGPUS --master_port 9992 \
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath $OUT_PATH/lavse/liwe_gru_gru/f30k_precomp.en/ \
--beta 0.991 \
--lr 6e-4 \
--workers 1 \
--text_encoder liwe_gru_gru \
--ngpu $NGPUS \
--vocab_path vocab/char.json \
--early_stop 100 \
--image_encoder hierarchical \
--text_repr liwe \
--text_pooling mean \


# 'lr': 6e-4,
#         'margin': 0.2,
#         'latent_size': 1024,
#         'grad_clip': 2.,
#         'workers': 2,
#         'text_encoder': 'liwe_gru',
#         'image_encoder': 'hierarchical',
#         'text_pooling': 'mean',
#         'text_repr': 'liwe',
#         'early_stop': 5,
#         'nb_epochs': 30,
#         'initial_k': 0.9,
#         'increase_k': 0.1,
#         'vocab_path': 'vocab/char.json',
