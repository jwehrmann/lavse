# Single dataset
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--profile liwe \
--outpath $OUT_PATH/lavse/liwe/f30k_precomp.en/


python train.py \
--data_path $DATA_PATH \
--train_data m30k_precomp.de \
--val_data m30k_precomp.de \
--profile liwe \
--outpath $OUT_PATH/lavse/liwe/m30k_precomp.de/


python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.de \
--val_data m30k_precomp.de \
--profile liwe \
--outpath $OUT_PATH/lavse/liwe/f30k_precomp.de/


python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.jt \
--val_data jap_precomp.jt \
--profile liwe \
--outpath $OUT_PATH/lavse/liwe/jap_precomp.jt/


python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.jp \
--val_data jap_precomp.jp \
--profile liwe \
--outpath $OUT_PATH/lavse/liwe/jap_precomp.jp/


python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en  \
--profile liwe \
--outpath $OUT_PATH/lavse/liwe/coco_precomp.en/ \
--lr_decay_interval 10 \

