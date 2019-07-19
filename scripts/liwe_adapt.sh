
# Pair datasets
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de \
--adapt_data m30k_precomp.en-de \
--profile liwe \
--outpath $OUT_PATH/lavse/liwe/f30k_precomp.en_m30k_precomp.de/



# Pair datasets
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de \
--adapt_data m30k_precomp.en-de \
--profile liwe_384 \
--outpath $OUT_PATH/lavse/liwe_384/f30k_precomp.en_m30k_precomp.de/


# Pair datasets
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de \
--adapt_data m30k_precomp.en-de \
--profile liwe_512 \
--outpath $OUT_PATH/lavse/liwe_512/f30k_precomp.en_m30k_precomp.de/


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
--profile liwe_384 \
--outpath $OUT_PATH/lavse/liwe_384/coco_precomp.en-jt/
--lr_decay_interval 10 \


# Coco Adapt
python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.en \
--val_data jap_precomp.en jap_precomp.jt \
--adapt_data jap_precomp.en-jt \
--profile liwe_384 \
--outpath $OUT_PATH/lavse/liwe_384/jap_precomp.en-jt/
--lr_decay_interval 10 \

