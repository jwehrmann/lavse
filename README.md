# Language-Agnostic Visual-Semantic Embeddings (LAVSE)

## Todo

- [] Liwe + Word emb (glove)
- [] Ensemble
- [] Lr_multiplier 
- [] Data Parallel 
- [] Distributed Data Parallel
- [] Fix `if master`
- [] Auto vocab cache
- [x] Avg Rich Embedding
- [x] Save yaml !
- [x] Test with yaml !
- [x] Batch and collate via dict
- [x] Optimizer Adamax 
- [x] Fix __include__ from yaml
- [x] Test via yaml
- [x] Fix t2i

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


