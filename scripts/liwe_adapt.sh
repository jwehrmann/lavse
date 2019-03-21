
# Pair datasets
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de \
--adapt_data m30k_precomp.en-de \
--profile liwe \
--outpath $OUT_PATH/lavse/liwe/f30k_precomp.en_m30k_precomp.de/


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en jap_precomp.jt \
--adapt_data jap_precomp.en-jt \
--profile liwe \
--outpath $OUT_PATH/lavse/liwe/f30k_precomp.en_jap_precomp.jt/


# All datasets
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de jap_precomp.jt \
--adapt_data m30k_precomp.en-de jap_precomp.en-jt \
--profile liwe \
--outpath $OUT_PATH/lavse/liwe/f30k_precomp.en_m30k_precomp.de_jap_precomp.jt/


# Coco Adapt
python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en jap_precomp.jt \
--adapt_data jap_precomp.en-jt \
--profile liwe \
--outpath $OUT_PATH/lavse/liwe/coco_precomp.en_jap_precomp.jt/

