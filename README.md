# Language-Agnostic Visual-Semantic Embeddings (LAVSE)

## Setup

```
export DATA_PATH=/opt/jonatas/datasets/lavse/
export OUT_PATH=/opt/jonatas/runs/lavse/
```

### Training Models

```
sh scripts/liwe_single.sh
```

### Evaluating Models 

```
python test.py \
--data_path $DATA_PATH \
--model_path $OUT_PATH/iccv/results/liwe_gru_384/coco_precomp.en/best_model.pkl \
--val_data coco_precomp.en \
--vocab_path vocab/char.json \
--text_repr liwe \
--outpath results/liwe_gru_384_coco_precomp.en.json
```


