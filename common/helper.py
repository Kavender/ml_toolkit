from typing import Iterable, Any, List, Generator
import time
import random


def delay_time(sleep_time = 160):
    time.sleep(sleep_time)


def stream(data: Iterable[Any], batch_size: int, shuffled: bool, n_batches: int = None, replace: bool = False) -> Generator[List[Any], None, None]:
    """
    Generates batches of elements from the input data.
    
    Parameters:
    - data (Iterable[Any]): The input data.
    - batch_size (int): The size of each batch.
    - shuffled (bool): Whether to shuffle the data before batching.
    - n_batches (int, optional): The number of batches to produce. Produces batches indefinitely if None. Default is None.
    - replace (bool, optional): Whether to sample with replacement. Default is False.
    
    Returns:
    - Generator[List[Any], None, None]: A generator yielding batches of elements.
    """
    # Convert iterable to list for indexing and shuffling
    data = list(data)
    if shuffled:
        random.shuffle(data)
    
    n = 0
    while n_batches is None or n < n_batches:
        if replace:
            batch = [random.choice(data) for _ in range(batch_size)]
        else:
            if batch_size > len(data):
                raise ValueError(f"batch_size ({batch_size}) cannot be greater than the length of data ({len(data)}) without replacement.")
            batch = random.sample(data, batch_size)
        yield batch
        n += 1


