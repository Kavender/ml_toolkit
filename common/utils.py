import os
import time
import pickle


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        # Python >2.5
        pass

def delay_time(sleep_time = 160):
    time.sleep(sleep_time)


def flatten_nested(lst):
    """Flatten a list with arbitrary levels of nesting.
    source: http://stackoverflow.com/questions/10823877/
        what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
    Args:
        lst (list): The nested list.
    Returns:
        (generator): The new flattened list of words.
    """
    if not isinstance(lst, list):
        yield []
    for i in lst:
        if any([isinstance(i, list), isinstance(i, tuple)]):
            for j in flatten(i):
                yield j
        else:
            yield i


def save2pickle(file_path, data):
    with open(file_path, 'wb') as infile:
        pickle.dump(data, infile, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_pickle(file_path):
    with open(file_path, 'rb') as outfile:
        data = pickle.load(outfile)
    return data

def save_processed_records(file_path, data):
    """save a dict of data one by one, in the order of processed"""
    if not os.path.exists(file_path):
        open(file_path, 'ab').close()
    with open(file_path, 'ab') as outfile:
        pickle.dump(data, outfile)
    outfile.close()

def load_processed_records(file_path):
    data = []
    try:
        with open(file_path, 'rb') as infile:
            while True:
                data.append(pickle.load(infile))
    except EOFError:
        pass
    return data
