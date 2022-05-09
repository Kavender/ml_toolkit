from numpy import array, corrcoef
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import anomaly_detection as detector
import smoothing as smoother
from helper import prepare_ts_for_detector
from visualize_trend import display_trend_with_breakpoints
from trend_utils import pairwise
from trend_analysis import identify_trend_and_breakpoints, analyze_trend_segment
import trend_correlation
from fbprophet import Prophet


if __name__ == "__main__":
    data_folder = os.path.join(os.getcwd(), 'data')
    data = pd.read_csv(data_folder+"/cars_with_segments.csv")
    data.set_index('Date', inplace=True)
    print("ts_data:", data.shape, data.columns)
    ts = data["gm_cap"] #"#Passengers"

    data = pd.read_csv(data_folder+"/cars_with_segments.csv")
    test_data = data[['Date', 'gm_cap']]
    test_data.columns = ['ds', 'y']
    print("test_Data", test_data.shape, test_data.tail(3))

    m = Prophet()
    m.fit(test_data)
    future = m.make_future_dataframe(periods=365)
    print(future.tail())
    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    fig1 = m.plot(forecast)
    plt.show(block=False)
    fig2 = m.plot_components(forecast)
    plt.show(block=False)

    from fbprophet.plot import add_changepoints_to_plot
    # m = Prophet(changepoint_range=0.6)
    # future = m.make_future_dataframe(periods=365)
    # forecast = m.predict(future)
    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)
    plt.show(block=True)
    exit()

    # ts_tesla = data["tesla_cap"]
    # ts_gm = data["gm_cap"]
    # print("Overall correlction", data.corr())
    # cm = corrcoef(data.values.T)
    # sns.set(font_scale=1.25)
    # cols = data.columns
    # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols, xticklabels=cols)
    # plt.show()

    # smoothed_ts_tesla = smoother.smoothing_localized_regression(ts_tesla)
    # smoothed_ts_gm = smoother.smoothing_localized_regression(ts_gm)
    # smoothed_ts_tesla, breaker_tesla = identify_trend_and_breakpoints(ts_tesla, search_method=detector.Binseg)
    # smoothed_ts_gm, breaker_gm = identify_trend_and_breakpoints(ts_gm, search_method=detector.Binseg)

    # print("cross correlation", trend_correlation.calculate_cross_correlation(smoothed_ts_tesla, ts_gm))
    # print("time shift btw two ts:", trend_correlation.calculate_shift(smoothed_ts_tesla, ts_gm))
    #
    # r, p = trend_correlation.calculate_pearson_correlation(smoothed_ts_tesla, ts_gm)
    # f,ax=plt.subplots(figsize=(10, 8))
    # ax.plot(smoothed_ts_tesla.index, smoothed_ts_tesla.values, label="tesla_cap")
    # ax.plot(smoothed_ts_gm.index, smoothed_ts_gm.values, label="ts_gm")
    # ax.set(xlabel='Time',ylabel='Pearson r')
    # plt.show()
    #
    # d, path = trend_correlation.calculate_dtw(smoothed_ts_tesla, ts_gm)
    # plt.plot(path[0], path[1], 'w')
    # plt.xlabel('ts1')
    # plt.ylabel('ts2')
    # plt.title(f'DTW Minimum Path with minimum distance: {round(d,2)}')
    # plt.show()
    # exit()
    ####
    ##analysis trend segment of one ts
    ts = data["gm_cap"] #"#Passengers"
    smoothed_decompose = smoother.smoothing_with_decomposition(ts,freq=350, model="additive")
    smoothed_ts, breaker_trend = identify_trend_and_breakpoints(ts, search_method=detector.Binseg)

    # exit()
    smoothed_ts_lowess = smoother.smoothing_localized_regression(ts)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(ts.index, ts.values, label="raw")
    ax.plot(ts.index, smoothed_ts_lowess.values, color = "g", label="LOWESS smoothed")
    ax.plot(ts.index, smoothed_decompose.values, color = "y", label="decomposed trend")
    ax.plot(ts.index, smoothed_ts.values, color = "r", label="combined smoothed")
    ax.legend(loc="upper left")
    plt.show(block=True)

    fig2, ax2 = display_trend_with_breakpoints(smoothed_ts, breaker_trend, "%y-%q")
    plt.title(f"Trend detection with Breakpoints")
    plt.show(block=True)

    for start, end in pairwise([0] + breaker_trend + [len(smoothed_ts)]):
        if end - start < 1:
            continue
        segment = smoothed_ts[start: end]
        segment_attr = analyze_trend_segment(segment)
        print((start, end), segment_attr)
    #######################
    # test_stationarity(ts) # verified, working

    # raw Detection
    # ts_data = prepare_ts_for_detector(ts)
    # print("check ts_data for anomoly detecter", len(ts_data))
    # anomolies = detector.detect_anomalies(ts_data, 5)
    # print("anomolies:", len(anomolies))
    # for anomoly in anomolies:
    #     print((anomoly.exact_timestamp, anomoly.anomaly_score), anomoly.get_time_window())

    # ts_decomposed = smoother.seasonal_decompose(ts, model="additive", freq=12)
    # print("ts_decomposed--seasonale:", ts_decomposed.seasonal.head(3))
    # ts = ts_decomposed.seasonal
