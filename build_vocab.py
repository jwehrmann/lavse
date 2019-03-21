import argparse
from lavse.utils.logger import create_logger
from lavse.data import get_loaders
from lavse.tokenizer import Tokenizer
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
    )
    parser.add_argument(
        '--char_level',
        action='store_true',
    )
    parser.add_argument(
        '--data_name',
        nargs='+',
        default=['f30k_precomp'],
    )
    parser.add_argument(
        '--outpath',
    )
    args = parser.parse_args()
    
    logger = create_logger(level='debug')

    data_path = Path(args.data_path)
    outpath = Path(args.outpath)
    
    files = []
    tokenizer = Tokenizer(
        download_tokenizer=True, 
        char_level=args.char_level
    )

    for data_name in args.data_name:
        data_name, lang = data_name.split('.')
        files.extend([
            data_path / data_name / f'train_caps.{lang}.txt',
            data_path / data_name / f'dev_caps.{lang}.txt',
        ])
    
    logger.info(f'Building vocab on {files}')
    tokenizer.fit_on_files(files)
    tokenizer.save(outpath)
    logger.info(f'Saved to {outpath}')
    logger.info('Done.')