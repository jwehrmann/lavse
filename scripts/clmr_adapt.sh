
# # Pair datasets
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de \
--adapt_data m30k_precomp.en-de \
--profile clmr \
--outpath $OUT_PATH/lavse/clmr/f30k_precomp.en_m30k_precomp.de/


# # Pair datasets
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de \
--adapt_data m30k_precomp.en-de \
--profile clmr_gru \
--outpath $OUT_PATH/lavse/clmr_gru/f30k_precomp.en_m30k_precomp.de/


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en jap_precomp.jt \
--adapt_data jap_precomp.en-jt \
--profile clmr \
--outpath $OUT_PATH/lavse/clmr/f30k_precomp.en_jap_precomp.jt/


python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en jap_precomp.jp \
--adapt_data jap_precomp.en-jp \
--profile clmr \
--outpath $OUT_PATH/lavse/clmr/f30k_precomp.en_jap_precomp.jp/


# All datasets
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en m30k_precomp.de jap_precomp.jt \
--adapt_data m30k_precomp.en-de jap_precomp.en-jt \
--profile clmr \
--outpath $OUT_PATH/lavse/clmr/f30k_precomp.en_m30k_precomp.de_jap_precomp.jt/


# Coco Adapt
python train.py \
--data_path $DATA_PATH \
--train_data coco_precomp.en \
--val_data coco_precomp.en jap_precomp.jt \
--adapt_data jap_precomp.en-jt \
--profile clmr \
--outpath $OUT_PATH/lavse/clmr/coco_precomp.en_jap_precomp.jt/

