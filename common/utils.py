import os
import time
import hashlib

# TODO: need a generic hash generator, support hashlib and pyhash
def hash_text(text: str)-> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_weeknum(dt):
    try:
        return int(dt.isocalender()[1])
    except:
        pass
    return None



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
