# export DATA_PATH=/home/jonatas/data/lavse/
# export OUT_PATH=runs/temp/

# Single dataset: 
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--profile clmr \
--outpath $OUT_PATH/lavse/clmr/f30k_precomp.en/ \
--eval_before_training


python train.py \
--data_path $DATA_PATH \
--train_data m30k_precomp.de \
--val_data m30k_precomp.de \
--profile clmr \
--outpath $OUT_PATH/lavse/clmr/m30k_precomp.de/


python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.jt \
--val_data jap_precomp.jt \
--profile clmr \
--outpath $OUT_PATH/lavse/clmr/jap_precomp.jt/


python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.jp \
--val_data jap_precomp.jp \
--profile clmr \
--outpath $OUT_PATH/lavse/clmr/jap_precomp.jp/


python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en  \
--profile clmr \
--outpath $OUT_PATH/lavse/clmr/coco_precomp.en/ \
--lr_decay_interval 10 \

