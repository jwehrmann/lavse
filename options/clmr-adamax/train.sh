python run.py -o options/clmr-adamax/de.yaml
python test.py options/clmr-adamax/de.yaml --data_split test

python run.py -o options/clmr-adamax/jt.yaml
python test.py options/clmr-adamax/jt.yaml --data_split test

python run.py -o options/clmr-adamax/f30k.yaml
python test.py options/clmr-adamax/f30k.yaml --data_split test

python run.py -o options/clmr-adamax/en-de.yaml
python test.py options/clmr-adamax/en-de.yaml --data_split test

python run.py -o options/clmr-adamax/en-jt.yaml
python test.py options/clmr-adamax/en-jt.yaml --data_split test

python run.py -o options/clmr-adamax/coco.yaml
python test.py options/clmr-adamax/coco.yaml --data_split test

python run.py -o options/clmr-adamax/coco-jt.yaml
python test.py options/clmr-adamax/coco-jt.yaml --data_split test

python run.py -o options/clmr-adamax/en-de-jt.yaml
python test.py options/clmr-adamax/en-de-jt.yaml --data_split test
