
python run.py -o options/scan/en-de.yaml
python test.py options/scan/en-de.yaml --data_split test

python run.py -o options/scan/en-jt.yaml
python test.py options/scan/en-jt.yaml --data_split test

python run.py -o options/scan/de.yaml
python test.py options/scan/de.yaml --data_split test

python run.py -o options/scan/jt.yaml
python test.py options/scan/jt.yaml --data_split test

python run.py -o options/scan/f30k.yaml
python test.py options/scan/f30k.yaml --data_split test

python run.py -o options/scan/coco.yaml
python test.py options/scan/coco.yaml --data_split test

