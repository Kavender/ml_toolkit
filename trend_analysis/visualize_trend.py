from typing import Tuple, List, Iterable
from itertools import cycle, tee
from datetime import datetime
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, datestr2num, num2date
from trend_utils import pairwise
from trend_globals import COLOR_CYCLE


def get_quarter(dt: datetime) -> str:
    return "Q%d" % math.ceil(dt.month/3)


def express_date(dt: datetime, fmt: str, tz=None):
    if isinstance(dt, datetime):
        pass
    elif isinstance(dt, np.datetime64):
        dt = pd.to_datetime(dt)
    elif isinstance(dt, str):
        dt = num2date(datestr2num(dt))
    else:
        raise ValueError("pls provide a date")

    fmt = re.sub(r"%Q", get_quarter(dt), fmt)
    fmt = re.sub(r"%q", get_quarter(dt).lower(), fmt)
    return DateFormatter(fmt=fmt, tz=tz).strftime(dt = dt)

# toDo: can be generalized to plot obs vs smoothed data (given smoothing function)
def plot_trend(ts: pd.Series, smoothed_ts: pd.Series, fitted_ts: List[np.array]) -> None:
    """
    Plot the macro trends extracted from data series

    Args:
        data: data series from which the breakpoints are analyzed
        fitted_trends: macro trends extracted
        smoothed_data: smoothed version of data

    """

    trend_dots = try_concatenate_fitted_trends(fitted_ts)

    fig, ax = plt.subplots()
    ax.plot(data.values, label=data.name, c='blue')
    ax.plot(trend_dots, label="trend", c="red")

    if not all(smoothed_ts.values == data.values):
        ax.plot(smoothed_ts, label="smoothed_ts")
    ax.legend(loc="upper left")

    xticks, labels = get_xticks_and_labels(ts)
    ax.set_ylim(bottom=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation="90")
    ax.xaxis.set_minor_locator(month)
    ax.xaxis.grid(True, which = 'minor')
    ax.xaxis.set_major_locator(year)
    ax.xaxis.set_major_formatter(year_format)
    plt.title("Visualize Trend Detection")


def estimate_xticker_steps(n):
    if 10 < n <= 20:
        return 2
    elif 20 < n <= 50:
        return 5
    elif 50 < n <= 100:
        return 10
    elif 100 < n <= 500:
        return 20
    elif 500 < n <= 100:
        return 50
    else:
        return 100


def get_xticks_and_labels(ts: pd.Series, ts_fmt: str) -> Tuple[list, list]:
    """
    For plotting, get xaxis ticks and labels based on index of input data series
    Args:
        data: data series indexed on time string

    Returns:
        list of xtick locations and labels
    """
    n = len(ts.index)

    if n <= 10:
        xticks = list(range(n))
        labels = [express_date(t, ts_fmt) for i, t in enumerate(ts.index)]
    else:
        step = estimate_xticker_steps(n)
        xticks = [i for i in range(n) if i % step == 0]
        labels = [express_date(t, ts_fmt) for i, t in enumerate(ts.index) if i % step == 0]
        if n % step != 0:
            xticks += [n - 1]
            labels += [express_date(ts.index[-1], ts_fmt)]
    return xticks, labels


def display_trend_with_breakpoints(ts: pd.Series, breakpoints: List[int], ts_fmt: str):
    """
    source code: http://ctruong.perso.math.cnrs.fr/ruptures-docs/build/html/_modules/ruptures/show/display.html#display
    Display a signal and the change points provided in alternating colors. If another set of change
    point is provided, they are displayed with dashed vertical dashed lines.

    Args:
        ts: data series indexed on time string
        breakpoints: list of change point indexes.

    Returns:
        tuple: (figure, ax) with a :class:`matplotlib.figure.Figure` object and an array of Axes objects.

    """
    xticks, labels = get_xticks_and_labels(ts, ts_fmt)

    fig, ax = plt.subplots(figsize=(15, 10))

    color_cycle = cycle(COLOR_CYCLE)
    ax.plot(range(len(ts.values)), ts.values, label=ts.name)

    # color each (true) regime
    bkps = [0] + sorted(breakpoints)

    for (start, end), col in zip(pairwise(bkps), color_cycle):
        ax.axvspan(max(0, start - 0.5), end - 0.5, facecolor=col, alpha=0.2)

    # add vertical lines to mark breakpoints
    for bkp in bkps:
        if bkp != 0 and bkp < len(ts):
            ax.axvline(x=bkp - 0.5, color="k", linewidth=3, linestyle="--")

    ax.set_ylim(bottom=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, rotation="90")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig, ax
