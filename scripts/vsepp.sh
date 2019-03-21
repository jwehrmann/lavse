# Single dataset
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--profile vsepp \
--outpath $OUT_PATH/lavse/vsepp/f30k_precomp.en/


python train.py \
--data_path $DATA_PATH \
--train_data m30k_precomp.de \
--val_data m30k_precomp.de \
--profile vsepp \
--outpath $OUT_PATH/lavse/vsepp/m30k_precomp.de/


python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.jt \
--val_data jap_precomp.jt \
--profile vsepp \
--outpath $OUT_PATH/lavse/vsepp/jap_precomp.jt/


python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.jp \
--val_data jap_precomp.jp \
--profile vsepp \
--outpath $OUT_PATH/lavse/vsepp/jap_precomp.jp/


python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en  \
--profile vsepp \
--outpath $OUT_PATH/lavse/vsepp/coco_precomp.en/ \
--lr_decay_interval 10 \

