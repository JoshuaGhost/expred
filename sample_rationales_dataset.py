import argparse
import json
import logging
import random
import shutil
import sys
from math import floor
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_valid_file(parser, arg):
    path = Path(arg)
    if not path.exists():
        parser.error("The file %s does not exist!" % arg)
    else:
        return path


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __str__(self):
        return '[{0},{1}]'.format(self.start, self.end)


copy_splits = ['val', 'test']
reduce_splits = ['train']


def sample_dataset(dataset_dir: Path, keep_rationals_fraction: float):
    logger.info(f'Sampling for keep fraction {keep_rationals_fraction} ')
    dataset_name = dataset_dir.stem
    datasets_root = dataset_dir.parent
    output_path: Path = datasets_root / f'{dataset_name}_{keep_rationals_fraction}'

    if output_path.exists():
        raise IOError(f'Output folder {output_path} already exits!')

    output_path.mkdir(parents=True, exist_ok=False)

    logger.info('Copying documents')
    # copy documents
    if (dataset_dir / 'docs').exists():
        shutil.copytree(dataset_dir / 'docs', output_path / 'docs')
    elif (dataset_dir / 'docs.jsonl').exists():
        shutil.copy(dataset_dir / 'docs.jsonl', output_path / 'docs.jsonl')
    else:
        raise ValueError(f'No documents found in {dataset_dir}')

    logger.info(f'Copying unchanged split {copy_splits}')
    # copy
    for split in copy_splits:
        file_name = f'{split}.jsonl'
        shutil.copy(dataset_dir / file_name, output_path / file_name)

    logger.info(f'Sampling and writing results for {reduce_splits}')
    # sample and copy
    for split in reduce_splits:
        file_name = f'{split}.jsonl'
        with open(dataset_dir / file_name, 'r') as f:
            annotations = [json.loads(line) for line in f]

        n = len(annotations)
        n_keep_rationales = floor(n * keep_rationals_fraction)
        indices_to_keep = random.sample(range(n), n_keep_rationales)

        def remove_rationales(annotation):
            annotation['evidences'] = []
            return annotation

        reduces_annotations = [
            ann if i in indices_to_keep else remove_rationales(ann)
            for i, ann in enumerate(annotations)
        ]

        with open(output_path / file_name, 'w') as f:
            dumped_annotations = map(json.dumps, reduces_annotations)
            f.write('\n'.join(dumped_annotations))


def main(args):
    parser = argparse.ArgumentParser(
        'Takes an eraser dataset and samples a given fraction of the annotations to keep the rationales and removes the rest')
    parser.add_argument('--dataset_dir', type=lambda x: is_valid_file(parser, x))
    parser.add_argument('--keep_rationals_fractions', nargs='+', type=float, choices=Range(0, 1))

    args = parser.parse_args(args)

    logger.info(
        f'Running sampling process for fraction {args.keep_rationals_fractions} for  dataset  {args.dataset_dir}')
    for fraction in args.keep_rationals_fractions:
        sample_dataset(args.dataset_dir, fraction)


if __name__ == '__main__':
    main(sys.argv[1:])
