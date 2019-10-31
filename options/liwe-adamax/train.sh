python run.py -o options/liwe-adamax/de.yaml
python test.py options/liwe-adamax/de.yaml --data_split test

python run.py -o options/liwe-adamax/f30k.yaml
python test.py options/liwe-adamax/f30k.yaml --data_split test

python run.py -o options/liwe-adamax/jt.yaml
python test.py options/liwe-adamax/jt.yaml --data_split test

python run.py -o options/liwe-adamax/en-de.yaml
python test.py options/liwe-adamax/en-de.yaml --data_split test

python run.py -o options/liwe-adamax/en-jt.yaml
python test.py options/liwe-adamax/en-jt.yaml --data_split test

python run.py -o options/liwe-adamax/coco.yaml
python test.py options/liwe-adamax/coco.yaml --data_split test

python run.py -o options/liwe-adamax/coco-jt.yaml
python test.py options/liwe-adamax/coco-jt.yaml --data_split test


