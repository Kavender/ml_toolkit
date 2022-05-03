from typing import Iterable
import itertools
import numpy as np

def prepare_ts_for_detector(ts):
    "transform ts series to dictionary for detector"
    return dict(zip(range(len(ts)), ts.values))


def map_ts_index_after_detection(ts):
    "store a location to ts index (datetime) mapping"
    return dict(zip(range(len(ts)), ts.index))


def calculate_pct_change(raw_change: float, value_from: float) -> float:
    """
    Calculate percent change using the amount changed and the original value

    Args:
        raw_change: amount changed from time_to to time_from
        value_from: value at time_from

    Returns:
        Percent change of raw_change from value_from
    """
    if value_from > 0:
        return raw_change * 100 / value_from
    if raw_change == 0:
        return np.nan
    return np.inf


def get_trend_change_type(pvalue: float, slope: float) -> str:
    """
    Helper function to determine the change type of a trend

    Args:
        pvalue: p-value of Ordinary Least Squares regression
        slope: slope of trend

    Returns:
        Either level, up, or down trend
    """
    if pvalue >= 0.2 or slope == 0:
        return "level trend"
    if slope > 0:
        return "up trend"
    return "down trend"
