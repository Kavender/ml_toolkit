from typing import Optional, List
import numpy as np
import pandas as pd
from copy import deepcopy
from seasonal import fit_trend
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import seasonal_decompose
from trend_globals import THRED_TS_LIMIT


def smoothing_with_decomposition(ts: pd.Series, freq: int, model: str = ['additive', 'multiplicative']) -> pd.Series:
    # free: control how much we want to to smooth, e.g with daily data series, and freq=350, we're smoothing one year data, or yearly trend
    # freq can be None if the series.index is DatetimeIndex e.g rdf.index = pd.DatetimeIndex(tdf['time'])
    # a series is thought to have four components, all series have a level and noise,
    # Level: The average value in the series.
    # Trend: The increasing or decreasing value in the series.
    # Seasonality: The repeating short-term cycle in the series.
    # Noise: The random variation in the series.
    # Additive model: y(t) = Level + Trend + Seasonality + Noise
    # Multiplicative model: y(t) = Level * Trend * Seasonality * Noise
    ts_decomposed = seasonal_decompose(ts, model=model, freq=freq)
    smoothed_series = ts_decomposed.trend
    return smoothed_series


def smoothing_localized_regression(ts: pd.Series, frac: float=0.05) -> pd.Series:
    "LOWESS smoothing, localized weighted regression"
    smoothed = lowess(ts.values, np.arange(len(ts)), frac=frac)[:, 1]
    return pd.Series(smoothed, index=ts.index)

# fit_trend_anomaly_free(data, seasonality_window, ptimes) this is our function
# when there is anomoly points, we shall have a higher level func to detect, break series by anomoly, fit sub trends, and aggregate
def smoothing_with_trend_fit(ts: pd.Series, period: Optional[int] = None, ptimes=2):
    """
    Apply trend_fit to series with more than 6 data point and assume no anomoly breakpoints.
    Arg:
        ts: pandas Series of time series of data
        period : number, seasonal periodicity, for filtering the trend.
        ptimes : number, multiple of period to use as smoothing window size.
                 The smaller the more flexible. e.g. [1: 3]
    Returns:
        np array of trends, either raw or fitted time series of data
    """
    trend = np.array(ts.values)

    if len(trend) > THRED_TS_LIMIT:
        try:
            trend = fit_trend(data=ts, kind="spline", period=period, ptimes=ptimes)
        except ValueError:
            print("Failed to fit LSQUnivariateSpline. Skip smoothing.")
    return pd.Series(trend, index=np.array(ts.index))


# toDo: rewrite the whoe function including the sub_function
def smooth_with_anomaly_replaced(ts: pd.Series, anomaly_detected: List[tuple]) -> pd.Series:
    smoothed_ts = deepcopy(ts)

    for loc in anomaly_detected:
        if loc[1] <= loc[0] + 1:
            smoothed_ts = replace_with_neighbor_avg(smoothed_ts, loc[0]) # rename/rewrite: replace with moving avg
        else:
            smoothed_ts = replace_with_line(smoothed_ts, loc[0], loc[1]) # rename/rewrite:replace with trend

    return smoothed_ts
