
python train.py \
--data_path $DATA_PATH \
--train_data jap_precomp.jt \
--val_data jap_precomp.en jap_precomp.jt \
--adapt_data jap_precomp.en-jt \
--profile scan \
--sim scan_i2t \
--outpath $OUT_PATH/lavse/scan_i2t/jap_precomp.en-jt/ \
--valid_interval 500 \
--workers 0 \
--vocab vocab/complete.json \
