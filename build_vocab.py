import argparse
from lavse.utils import create_logger
from lavse.data import get_loaders
from lavse.tokenizers import WordTokenizer
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
    )
    parser.add_argument(
        '--data_name',
        nargs='+',
        default=['f30k_precomp'],
    )
    parser.add_argument(
        '--langs',
        nargs='+',
        default=['en'],
    )
    parser.add_argument(
        '--outpath',
    )
    args = parser.parse_args()
    
    logger = create_logger(level='info')

    data_path = Path(args.data_path)
    outpath = Path(args.outpath)
    
    files = []
    tokenizer = WordTokenizer(download_tokenizer=True)
    for data_name in args.data_name:
        for lang in args.langs:
            files.extend([
                data_path / data_name / f'train_caps.{lang}.txt',
                data_path / data_name / f'dev_caps.{lang}.txt',
            ])
    
    logger.info(f'Building vocab on {files}')
    tokenizer.fit_on_files(files)
    tokenizer.save(outpath)
    logger.info('Done.')