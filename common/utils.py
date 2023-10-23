from typing import List, Iterable
import os
import errno
import hashlib
from collections import OrderedDict


def ensure_path_exists(file_path):
    try:
        os.makedirs(file_path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def hash_text(text: str, algo: str = 'sha256')-> str:
    try:
        hash_func = getattr(hashlib, algo)
        return hash_func(text.encode('utf-8')).hexdigest()
    except AttributeError:
        raise ValueError(f"'{algo}' is not a supported algorithm in hashlib.")
    


def get_weeknum(dt):
    try:
        return int(dt.isocalender()[1])
    except:
        pass
    return None


def flatten_nested(lst: Iterable) -> Iterable:
    """
    Flatten an iterable with arbitrary levels of nesting.
    source: http://stackoverflow.com/questions/10823877/
        what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python
    
    Args:
        lst (Iterable): The nested iterable (e.g., list or tuple).
        
    Yields:
        item: Items from the flattened iterable.
        
    Examples:
        >>> list(flatten_nested([1, [2, 3], [4, [5, 6], 7], 8]))
        [1, 2, 3, 4, 5, 6, 7, 8]
        
        >>> list(flatten_nested([1, (2, 3), [[4, 5], (6, 7)], 8]))
        [1, 2, 3, 4, 5, 6, 7, 8]
    """
    # Check if lst is not an iterable or is a string, in which case yield the item itself
    if not isinstance(lst, Iterable) or isinstance(lst, (str, bytes)):
        yield lst
        return
    
    for i in lst:
        # Check if an item is an iterable (but not a string)
        if isinstance(i, Iterable) and not isinstance(i, (str, bytes)):
            for j in flatten_nested(i):
                yield j
        else:
            yield i


def as_ordered_dict(raw_dict, preference_orders: List[List[str]]) -> OrderedDict:
    """
    Returns Ordered Dict of Params from list of partial order preferences.
    """
    if len(raw_dict) != len(preference_orders) or len(set(raw_dict.keys()).symmetric_difference(preference_orders)) > 0:
        raise VauleError(f"Preferred order doesn't apply to the same of raw_dict: {raw_dict.keys()}")

    def order_func(key):
        # Makes a tuple to use for ordering.  The tuple is an index into each of the `preference_orders`,
        # followed by the key itself.  This gives us integer sorting if you have a key in one of the
        # `preference_orders`, followed by alphabetical ordering if not.
        order_tuple = [
            order.index(key) if key in order else len(order) for order in preference_orders
        ]
        return order_tuple + [key]

    def order_dict(dictionary, order_func):
        # Recursively orders dictionary according to scoring order_func
        result = OrderedDict()
        for key, val in sorted(dictionary.items(), key=lambda item: order_func(item[0])):
            result[key] = order_dict(val, order_func) if isinstance(val, dict) else val
        return result

    return order_dict(raw_dict, order_func)
