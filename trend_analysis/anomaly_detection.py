from typing import Dict, List, Optional, Any
from numpy import log, array, log
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from luminol.anomaly_detector import AnomalyDetector
from luminol.modules.anomaly import Anomaly
from ruptures.base import BaseEstimator
from ruptures import Binseg, BottomUp, Window
from trend_utils import consecutive, pairwise
from trend_globals import MIN_RATIO_OUTLIER, NOISE_STD


def test_stationarity(ts_series: pd.Series):
    """check stationary of time series if given statistical properties:
        1. constant mean, 2. constant variance 3.an autocovariance that does not depend on time
    """
    #Determing rolling statistics
    rolmean = ts_series.rolling(window=12).mean()
    rolstd = ts_series.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(ts_series, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=True)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(ts_series, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


def detect_anomalies(ts_data: Dict[int, float], score_threshold: float) -> List[Anomaly]:
    """
    Args:
        data: dict of data with key-value pairs of location and value
        score_threshold: threshold for determining anomalies

    Methods:
        AnomalyDetector class has the following public methods, https://github.com/linkedin/luminol
        get_all_scores(): returns an anomaly score time series of type TimeSeries.
        get_anomalies(): return a list of Anomaly objects.

    Returns:
        list of anomaly objects
    """

    detector = AnomalyDetector(time_series=ts_data, algorithm_name="derivative_detector",
                               score_threshold=score_threshold)
    anomalies = detector.get_anomalies()
    for anomaly in anomalies:
        anomaly.exact_timestamp = adjust_anomaly_pos(anomaly.exact_timestamp, ts_data)
    return anomalies


def adjust_anomaly_pos(bkp_loc: int, ts_data: Dict[int, Any]) -> int:
    """
    Depending on whether it's an outlier or a level shift, return the raw loc or the previous loc

    Args:
        raw_bkp_loc: integer value of location
        data_dict: dictionary of loc/value pairs

    Returns:
        either previous/next loc or raw_bkp_loc
    """
    if bkp_loc == 0:
        return bkp_loc + 1
    if bkp_loc == len(ts_data) - 1:
        return bkp_loc - 1

    curr_interval_change = ts_data[bkp_loc] - ts_data[bkp_loc - 1]
    next_interval_change = ts_data[bkp_loc + 1] - ts_data[bkp_loc]

    if abs(next_interval_change) / abs(curr_interval_change) >= MIN_RATIO_OUTLIER:
        return bkp_loc + 1
    # level shift
    else:
        return bkp_loc


def detect_breakpoints(trend: array, search_method: BaseEstimator) -> List[int]:
    """Detect breakpoints of trend fitted, with cost function detects changes in the median of a signal.

    Parameters
    ----------
    trend :
        The fitted trend data.

    Returns
    -------
    List[int]
        Location of breakpoints detected in time series data.

    """
    if len(trend) < 3:
        raise ValueError("Not enough data point to fit trend.")
    bkp_segments = detect_trend_segments(trend, search_method)

    bkp_locs = deepcopy(bkp_segments)
    for start, end in pairwise([0] + bkp_segments + [len(trend)]):
        trend_segment = trend[start: end]
        if len(trend_segment) < 3:
            continue
        bkp_locs_slope_changes = detect_change_of_slopes(trend_segment)
        bkp_locs = merge_neighbor_bkps(bkp_locs, bkp_locs_slope_changes)
    if bkp_locs and bkp_locs[-1] == len(trend):
        bkp_locs.pop()
    return bkp_locs


def merge_neighbor_bkps(bkp_source1, bkp_source2):
    """
    Update the breakpoints by removing duplicate location or those are consecutive.
    """
    sorted_bkps = sorted(set(bkp_source1).union(bkp_source2))
    return [bkp_group[0] for bkp_group in consecutive(sorted_bkps)]


def detect_change_of_slopes(trend: array) -> List[str]:
    """Detect breakpoints of trend fitted, if the slope of trend flips.

    Parameters
    ----------
    trend : array
        The fitted trend data.

    Returns
    -------
    List[str]
        Location of breakpoints detected in time series data..
    """
    bkp_locs = []
    diffs = [j - i for i, j in zip(trend[: -1], trend[1 :])]
    prev_diff = diffs[0]

    for idx, curr_diff in enumerate(diffs[1:]):
        if (prev_diff != 0 and (curr_diff/prev_diff) < 0) or (prev_diff == 0 and curr_diff != 0):
            bkp_locs.append(idx + 1)
        prev_diff = curr_diff
    return bkp_locs


def detect_trend_segments(trend: array, search_method: BaseEstimator, model: Optional[str] = None,
                          penalty: Optional[str] = 'epsilon'
                          ) -> List[int]:
    """Detect segment of trend with Binary Segementation, BottomUp or Window-based change point detection.

    Parameters
    ----------
    trend : array
        The fitted trend data.
    model: segment model, [“l1”, “l2”, “rbf”] as "linear", "normal", "ar".
    search_method: module implements the change point detection methods by ruptures.detection.
            [Binseg, BottomUp, Window]
    penalty: In the situation in which the number of change points is unknown, one can specify a penalty using:
             the 'pen' parameter or a threshold on the residual norm using 'epsilon'.

    Returns
    -------
    List[str]
        Location of breakpoints detected across segment of data.
    """
    if not model:
        model = "rbf"
    if not search_method:
        search_method = Window
    dim = 1 if len(trend.shape) == 1 else trend.shape[-1]
    # print("trend:", len(trend), len(trend.shape), trend[0:10])
    algo = search_method(model=model, min_size=3).fit(trend)
    if penalty == "pen":
        bkp_locs = algo.predict(pen=log(len(trend))*dim*NOISE_STD**2)
    elif penalty == "epsilon":
        bkp_locs = algo.predict(epsilon=3*len(trend)*NOISE_STD**2)
    else:
        raise ValueError("Invalid penalty applied in the situation in which the number of change points is unknown.")
    return bkp_locs
