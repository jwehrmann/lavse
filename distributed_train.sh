python -m torch.distributed.launch \
--nproc_per_node=$2 run.py -o $1 \
--ngpu $2 --misc.distributed true
