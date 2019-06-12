# Single dataset

python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.jt \
--val_data jap_precomp.jt \
--profile vse \
--outpath $OUT_PATH/lavse/vse/jap_precomp.jt/


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--profile vse \
--outpath $OUT_PATH/lavse/vse/f30k_precomp.en/


python train.py \
--data_path $DATA_PATH \
--train_data m30k_precomp.de \
--val_data m30k_precomp.de \
--profile vse \
--outpath $OUT_PATH/lavse/vse/m30k_precomp.de/


python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.jp \
--val_data jap_precomp.jp \
--profile vse \
--outpath $OUT_PATH/lavse/vse/jap_precomp.jp/


python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en  \
--profile vse \
--outpath $OUT_PATH/lavse/vse/coco_precomp.en/ \
--lr_decay_interval 10 \

