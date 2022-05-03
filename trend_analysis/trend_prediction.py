# https://github.com/WillKoehrsen/Data-Analysis/blob/master/additive_models/Additive%20Models%20for%20Prediction.ipynb
from numpy import array
import statsmodels.api as sm
from trend_globals import THRED_OLS_LIMIT

def fit_ols_trend(ts_values: array, inspect: bool=False):
    X = sm.add_constant(array(range(len(ts_values))))
    y = ts_values
    model = sm.OLS(y, X)
    model_fitted = model.fit()
    if inspect:
        print(result.summary())
    return model_fitted


def calculate_ols_trend_slope(ts_values: array):
    if len(ts_values) > THRED_OLS_LIMIT:
        fitted = fit_ols_trend(ts_values)
        slope = fitted.params[1]
        pvalue = fitted.pvalues[1]
    else:
        slope = (ts_values[-1] - ts_values[0]) / len(ts_values) - 1
        pvalue = 0
    return slope, pvalue
