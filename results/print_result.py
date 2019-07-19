import sys
sys.path.append('../')

from lavse.utils.file_utils import load_json

files = sys.argv[1:]

metrics = [
     'i2t_r1', 'i2t_r5', 'i2t_r10', 'i2t_meanr', 'i2t_medr',
     't2i_r1', 't2i_r5', 't2i_r10', 't2i_meanr', 't2i_medr',
]


def load_and_filter_file(file_path):

    result = load_json(file)
    result_filtered = {
        k.split('/')[1]: v
        for k, v in result.items()
        if k.split('/')[1] in metrics
    }

    res_line = '\t'.join(
            [f'{result_filtered[metric]:3.2f}' for metric in metrics]
        )
    _file = file_path.split('/')[-1]
    print(
        f'{_file:40s}\t{res_line}'
    )

for file in files:
    load_and_filter_file(file)
