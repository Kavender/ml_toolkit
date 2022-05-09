import scipy.stats as stats
from dtw import accelerated_dtw
from numpy import flipud, real, argmax
from numpy.fft import fft, ifft, fftshift
# When attempting to detect cross-correlation between two time series, the first thing you should do is make sure the time series are stationary (i.e. have a constant mean, variance, and autocorrelation).

def calculate_pearson_correlation(ts_1, ts_2):
    """Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
    Two data samples needs to have the same length."""
    corr, pvalue = stats.pearsonr(ts_1, ts_2)
    print(f"Scipy computed Pearson r: {corr} and p-value: {pvalue}")
    return corr, pvalue


def calculate_spearman_correlation(ts_1, ts_2):
    """Spearman's correlation coefficient = covariance(rank(X), rank(Y)) / (stdv(rank(X)) * stdv(rank(Y)))
    Two data samples needs to have the same length"""
    corr, pvalue = spearmanr(ts_1, ts_2)
    print(f"Scipy computed Spearman r: {corr} and p-value: {pvalue}")
    return corr, pvalue


def calculate_dtw(ts_1, ts_2):
    """DTW computes the path between two signals that minimize the distance between the two signals.
    One downside is that it cannot deal with missing values.
    """
    min_path_cost, cost_matrix, acc_cost_matrix, path = accelerated_dtw(ts_1.values, ts_2.values, dist='euclidean')
    return min_path_cost, path


def calculate_cross_correlation(ts_1, ts_2):
    f1 = fft(ts_1)
    f2 = fft(flipud(ts_2))
    cc = real(ifft(f1 * f2))
    return fftshift(cc)


def calculate_shift(ts_1, ts_2):
    "shift: time-delay offset between two time series"
    assert len(ts_1) == len(ts_2)
    cc = calculate_cross_correlation(ts_1, ts_2)
    assert len(cc) == len(ts_1)
    zero_index = int(len(ts_1) / 2) - 1
    shift = zero_index - argmax(cc)
    return shift
