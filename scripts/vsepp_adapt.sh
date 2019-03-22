# All datasets
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de jap_precomp.jt \
--adapt_data m30k_precomp.en-de jap_precomp.en-jt \
--profile vsepp \
--outpath $OUT_PATH/lavse/vsepp/f30k_precomp.en_m30k_precomp.de_jap_precomp.jt/


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en jap_precomp.jt \
--adapt_data jap_precomp.en-jt \
--profile vsepp \
--outpath $OUT_PATH/lavse/vsepp/f30k_precomp.en_jap_precomp.jt/


# Pair datasets
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de \
--adapt_data m30k_precomp.en-de \
--profile vsepp \
--outpath $OUT_PATH/lavse/vsepp/f30k_precomp.en_m30k_precomp.de/


# Coco Adapt
python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en jap_precomp.jt \
--adapt_data jap_precomp.en-jt \
--profile vsepp \
--outpath $OUT_PATH/lavse/vsepp/coco_precomp.en_jap_precomp.jt/
--lr_decay_interval 10 \

