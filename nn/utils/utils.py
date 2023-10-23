from typing import List, Dict, Union, Any, Iterable, TypeVar, Generator
import random
import numpy as np
from argparse import Namespace
from transformers import TrainingArguments
import torch

T = TypeVar('T')
SEED_NUM = 123


def str2bool(v):
    """Utility function for parsing boolean in argparse
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    :param v: value of the argument
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def set_seed(args: Union[Dict[str, Any], Namespace, TrainingArguments]):
    if isinstance(args, Namespace):
        args = vars(args)
    elif isinstance(args, TrainingArguments):
        args = args.to_dict()

    seed = args.get("seed", SEED_NUM)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # set all GPUs with the same seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def overlapping_batches(iterable: Iterable[T], max_batch_length: int, overlap: int = 0) -> Generator[List[T], None, None]:
    """
    Batch data from an iterable into slices of a specified length. The last batch may be shorter.
    
    Parameters:
    - iterable (Iterable): The input iterable to be batched.
    - max_batch_length (int): Maximum number of items in each batch.
    - overlap (int, optional): Number of items that will overlap between batches. Default is 0.
    
    Yields:
    - List: Batches of items from the input iterable.
    
    Examples:
    >>> list(overlapping_batches("ABCDEFG", 3))
    ['ABC', 'DEF', 'G']
    >>> list(overlapping_batches("ABCDEFG", 3, overlap=1))
    ['ABC', 'CDE', 'EFG', 'G']
    """
    if max_batch_length < 1:
        raise ValueError("max_batch_length must be at least one")
    if overlap >= max_batch_length:
        raise ValueError("overlap must be smaller than max_batch_length")
    
    for i in range(0, len(iterable), max_batch_length - overlap):
        yield iterable[i : i + max_batch_length]


def get_minibatch(X, y, minibatch_size):
    minibatches = []

    X, y = random.shuffle(X, y)

    for i in range(0, X.shape[0], minibatch_size):
        X_mini = X[i:i + minibatch_size]
        y_mini = y[i:i + minibatch_size]

        minibatches.append((X_mini, y_mini))

    return minibatches


if __name__ == '__main__':
    import doctest
    doctest.testmod()