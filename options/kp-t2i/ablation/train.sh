python run.py -o options/kp-t2i/ablation/f30k.p256.yaml
python test.py options/kp-t2i/ablation/f30k.p256.yaml --data_split test
python run.py -o options/kp-t2i/ablation/f30k.p128.yaml
python test.py options/kp-t2i/ablation/f30k.p128.yaml --data_split test
python run.py -o options/kp-t2i/ablation/f30k.p64.yaml
python test.py options/kp-t2i/ablation/f30k.p64.yaml --data_split test
python run.py -o options/kp-t2i/ablation/f30k.p32.yaml
python test.py options/kp-t2i/ablation/f30k.p32.yaml --data_split test




python run.py -o options/kp-t2i/ablation/f30k.p16.yaml
python test.py options/kp-t2i/ablation/f30k.p16.yaml --data_split test
python run.py -o options/kp-t2i/ablation/f30k.p8.yaml
python test.py options/kp-t2i/ablation/f30k.p8.yaml --data_split test
python run.py -o options/kp-t2i/ablation/f30k.p4.yaml
python test.py options/kp-t2i/ablation/f30k.p4.yaml --data_split test
python run.py -o options/kp-t2i/ablation/freeze.yaml
python test.py options/kp-t2i/ablation/freeze.yaml --data_split test
python run.py -o options/kp-t2i/ablation/train_gamma.yaml
python test.py options/kp-t2i/ablation/train_gamma.yaml --data_split test
