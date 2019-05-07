
python train.py \
--data_path $DATA_PATH \
--outpath $OUT_PATH/temp/dynamic_i2t/f30k_precomp.en/ \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--workers 3 \
--sim rnn_proj \
--image_encoder hierarchical \
--text_encoder emb_proj \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--vocab vocab/f30k_vocab.json \
--valid_interval 500
