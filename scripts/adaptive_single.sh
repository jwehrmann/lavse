
python train.py \
--data_path $DATA_PATH \
--train_data f30k_precomp.en \
--val_data f30k_precomp.en \
--outpath runs/adaptive_t2i/f30k_precomp.en/ \
--sim adaptive \
--workers 2 \
--image_encoder hierarchical \
--text_encoder convgru_sa \
--text_pooling none \
--image_pooling none \
--lr 6e-4 \
--beta 0.999 \
--eval_before_training \
--vocab vocab/f30k_vocab.json \
--valid_interval 500
