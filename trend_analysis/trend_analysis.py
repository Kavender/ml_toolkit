from typing import Tuple, List, Dict, Optional, Any
import logging
import numpy as np
import pandas as pd
from anomaly_detection import detect_anomalies, detect_breakpoints, BaseEstimator, Anomaly
from smoothing import smoothing_with_trend_fit
from helper import prepare_ts_for_detector, calculate_pct_change, get_trend_change_type
from trend_prediction import fit_ols_trend, calculate_ols_trend_slope
from trend_globals import MIN_ANOMALY_SCORE, TS_DETECTOR_MODE
#https://github.com/alvarobartt/trendet
# https://facebook.github.io/prophet/docs/quick_start.html


def locate_lastest_anomaly(ts_data: Dict[int, float], anomalies: List[Anomaly]) -> Anomaly:
    """In order to not overly react to the short anomaly change at the end, add limit for the latest period to be at
    least five datapoint, or represent 5% of whole time series data."""
    length_ts = len(ts_data)
    ts_values = list(ts_data.values())
    if len(anomalies) == 1:
        return anomalies[0]
    lst_anomaly = anomalies[-1]
    lst_slope, _ = calculate_ols_trend_slope(ts_values[lst_anomaly.exact_timestamp:])
    for anomaly in anomalies[-2::-1]:
        pos_anomaly = anomaly.exact_timestamp
        curr_slope, _ = calculate_ols_trend_slope(ts_values[pos_anomaly:lst_anomaly.exact_timestamp])
        # print("range and slop:", (curr_slope, lst_slope), (pos_anomaly, lst_anomaly.exact_timestamp))
        if (length_ts - pos_anomaly) >= 5 or ((length_ts-pos_anomaly)/length_ts) >= 0.05:
            if (curr_slope * lst_slope) < 0 or abs(lst_slope) <= 0.001:
                return lst_anomaly
        lst_anomaly = anomaly
    return lst_anomaly


# toDo: we could add adj_earliest to reflact starting of trend as well
def identify_trend_and_breakpoints(ts_series: pd.Series, search_method: BaseEstimator, adj_latest: bool=True, mode:
                                   Optional[int]=TS_DETECTOR_MODE.SENSITIVE.value) -> Tuple[pd.Series, List[int]]:
    """
    This function receives as input a pandas.Series from which data is going to be analysed in order to
    detect/identify trends over a certain date range.
    this function will identify trend segments with breakpoints, keeping just the longer trend and discarding
    the nested trend.
    if adj_latest, we will sepearate the last trend segment before smoothing the whole overall trend, to make it refelct
    the latest trend movement.
    """
    ts_data = prepare_ts_for_detector(ts_series)
    anomalies = detect_anomalies(ts_data, MIN_ANOMALY_SCORE)

    if adj_latest and anomalies:
        latest_anomaly = locate_lastest_anomaly(ts_data, anomalies)
        print("check where the latest anomaly being found:", len(ts_data), latest_anomaly, anomalies[-1].start_timestamp, anomalies[-1].end_timestamp)
        splited_trends = [smoothing_with_trend_fit(ts_series[: latest_anomaly.exact_timestamp], ptimes=mode),
                          smoothing_with_trend_fit(ts_series[latest_anomaly.exact_timestamp: ],
                                                   ptimes=TS_DETECTOR_MODE.ROBUST.value)]
        smoothed_ts = pd.concat(splited_trends)
    else:
        smoothed_ts = smoothing_with_trend_fit(ts_series, ptimes=mode)
    print("smoothed ts:", len(smoothed_ts), type(smoothed_ts), smoothed_ts.head(3))
    bkp_smoothed = detect_breakpoints(smoothed_ts.values, search_method=search_method)
    print("raw_breaker points:", bkp_smoothed)
    return smoothed_ts, bkp_smoothed


def analyze_trend_segment(segment_series: pd.Series) -> Dict[str, Any]:
    """
    Analyze a single trend segment defined as the data series between two breakpoints,
    and characterize the segment with the following attributes:

    Args:
        segment_series: pd.Series, subset of time series trend

    Returns:
        dict for single trend insight attributes
    """
    ts_values = segment_series.values
    avg = np.mean(ts_values)
    pct_variation = np.std(ts_values) * 100 / abs(avg)

    slope, pvalue = calculate_ols_trend_slope(ts_values)
    avg_pct_change = calculate_pct_change(slope, ts_values[0])
    change_type = get_trend_change_type(pvalue, slope)

    return {"slope": slope, "avg_pct_change": avg_pct_change, "pct_variation": pct_variation,
            "change_type": change_type}
