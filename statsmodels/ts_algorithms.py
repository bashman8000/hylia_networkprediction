import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pandas.tools.plotting import autocorrelation_plot
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from sklearn import preprocessing
import numpy as np
from pandas.compat import lmap

def autocorrelation_plot(series, label, lower_lim=1, n_samples=None, ax=None, **kwds):
    """Autocorrelation plot for time series.

    Parameters:
    -----------
    series: Time series
    ax: Matplotlib axis object, optional
    kwds : keywords
        Options to pass to matplotlib plotting method

    Returns:
    -----------
    ax: Matplotlib axis object
    """
    import matplotlib.pyplot as plt
    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca(xlim=(lower_lim, n_samples), ylim=(-1.0, 1.0))
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[:n - h] - mean) *
                (data[h:] - mean)).sum() / float(n) / c0
    x = (np.arange(n) + 1).astype(int)
    y = lmap(r, x)
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / np.sqrt(n), linestyle='--', color='grey')
    ax.axhline(y=z95 / np.sqrt(n), color='grey')
    ax.axhline(y=0.0, color='black')
    ax.axhline(y=-z95 / np.sqrt(n), color='grey')
    ax.axhline(y=-z99 / np.sqrt(n), linestyle='--', color='grey')
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    if n_samples:
        ax.plot(x[:n_samples], y[:n_samples], label=label, **kwds)
    else:
        ax.plot(x, y, label=label, **kwds)
    if 'label' in kwds:
        ax.legend()
    ax.grid()
    return ax

#difference between timedeltas, in hours
def td_diff_hrs(td_other, td_benchmark):
    diff = td_other - td_benchmark
    hrs = diff.days*24 + diff.seconds//3600
    return hrs

#---Convert Datetime index to hours from start---
def index_to_hrs(series):
    start_hour = series.index[0]
    new_index = [td_diff_hrs(i, start_hour) for i in series.index]
    return pd.Series(data=series.values, index=new_index)

#---MSE---
#oh this is for multiple values
def mse(actual, pred):
    #normalize the data
    #don't need to do this if we are doing it from the start
    # actual = normalize(actual)
    # pred = normalize(pred)
    total = 0
    n = len(actual)
    for i in range(n):
        diff = actual[i] - pred[i]
        total += diff**2
    return float(total)/n

def normalize(data):
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    data = [[d] for d in data]
    scaler.fit(data)
    data = scaler.transform(data)
    data = [j for i in data for j in i]
    return data

def mse_upto_hour(actual, pred):
    mses = {1:0, 4:0, 8:0, 16:0, 24:0}
    for hr in mses.keys():
        mses[hr] = mse(actual[:hr], pred[:hr])
    return mses

def avg_link_mses(pairs):
    dicts = []
    for pair in pairs:
        dicts.append(mse_upto_hour(pair[0], pair[1]))
    df = pd.DataFrame(dicts)
    answer = dict(df.mean())
    return answer

def something(data):
    val_data = data[2152:]
    training_sizes = [50,100, 200, 500, 1000, 2000]
    arima_mse = {}
    hw_mse = {}
    for size in training_sizes:
        train_data = data[2152-size:2152]
        hw_pred = triple_exponential_smoothing(train_data, 24, 0.7, 0.02, 0.9, 8)
        hw_pred = hw_pred[len(hw_pred)-8:]
        arima_pred = arima(train_data, 8)
        hw_mse[size] = mse(val_data, hw_pred)
        arima_mse[size] = mse(val_data, arima_pred)
    return hw_mse, arima_mse


#---Impute Missing Data---
def fill_missing(series):
    nulls = series.isnull().values
    prev_not_null = -1
    impute_indices = []
    for i in range(len(series)):
        curr = series[i]
        curr_is_null = nulls[i]
        if(curr_is_null):
            impute_indices.append(i)
            #if this is the last element
            #then set all values at indices in array to prev_not_null
            #will error if the entire dataset is null
            if(i==len(series)-1):
                for j in impute_indices:
                    series[j] = prev_not_null
        else:
            if(len(impute_indices)!=0):
                #if the array contains the first element
                #then set all values at indices in array to curr
                #will error if the entire dataset is null
                if(0 in impute_indices):
                    fill_value = curr
                else:
                    fill_value = (curr + prev_not_null) /2.0
                for j in impute_indices:
                    series[j] = fill_value
                #once they've all been filled, reset
                impute_indices = []
            #if curr is not null,
            #the previous not null value is curr
            prev_not_null = curr
    return series

#---Convert list to series with DateTime Index--
#there's definitely a better way to do this...
def preds_to_series(og, preds):
    copy = og.copy(deep=True)
    for i in range(len(copy)):
        copy[i] = preds[i]
    return copy

#---Plot actual and predicted---
#labeled_data is a dict of {label:data}
def plot(labeled_data, title):
    fig = plt.figure()
    for label in labeled_data.keys():
        data = labeled_data[label]
        plt.plot(data, label=label, linewidth=0.7)
    #plt.xticks([2135,2159],["2136", "2160"])
    plt.legend()
    plt.title(title)
    plt.show()

#---Holt Winters---
#take 2 seasons, subtract their corresponding elements, add them all up, and divide by slen^2
def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i])
    return sum / (slen*slen)

def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        numer = sum(series[slen*j:slen*j+slen])
        avg = numer/float(slen)
        season_averages.append(avg)
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result

#---ARIMA---
#p,d,q= 1,0,0
def arima(data, n_preds):
    model = ARIMA(data, order=(1,0,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(steps=n_preds)
    arima_predictions = output[0]
    return arima_predictions




