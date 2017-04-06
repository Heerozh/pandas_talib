"""
ta-lib tested pure pandas implement

Repository:
https://github.com/Heerozh/pandas_talib
"""
import pandas as pd
import numpy as np

__all__ = ['MA', 'SMA', 'EMA', 'STDDEV', 'MOM', 'ROC', 'ROCP', 'ATR', 'BBANDS', 'MACD', 'RSI']


def out(df, result, join, dropna):
    if join:
        result = df.join(result, how='inner')

    if dropna:
        result.dropna(inplace=True)
    return result


def sel_columns(df, columns, new_names):
    if type(columns) is str:
        columns = [columns]

    if type(df) is pd.Series:
        result = df
    else:
        result = df[columns]

    if new_names:
        return result.rename(columns=dict(zip(columns, new_names)))
    else:
        return result


def join_result(dfs, columns, new_names):
    if type(columns) is str:
        columns = [columns]

    if new_names is None or new_names is False:
        join = np.array([[s + str(i) for i in range(len(dfs))] for s in columns])
    else:
        join = np.array(new_names)

    result = dfs[0]
    for i in range(len(dfs)):
        dfs[i].rename(columns=dict(zip(columns, join[:, i])), inplace=True)
        if i > 0:
            result = result.join(dfs[i], how='inner')
    return result


def MA(df, columns, n, join=None, dropna=True):
    """
    Moving Average
    """
    result = sel_columns(df, columns, join)
    result = result.rolling(n).mean()
    return out(df, result, join, dropna)


SMA = MA


def EMA(df, columns, n, join=None, dropna=True, min_periods=0):
    """
    Exponential Moving Average
    """
    result = sel_columns(df, columns, join).copy()
    result.iloc[:n] = np.mean(result.values[:n])
    # print(n, result)
    result = result.ewm(span=n, min_periods=min_periods, adjust=False).mean()
    return out(df, result, join, dropna)


def STDDEV(df, columns, n, join=None, dropna=True):
    """
    Standard Deviation
    """
    result = sel_columns(df, columns, join)
    result = result.rolling(n).std(ddof=0)
    return out(df, result, join, dropna)


def MOM(df, columns, n, join=None, dropna=True):
    """
    Momentum
    """
    result = sel_columns(df, columns, join)
    result = result.diff(n)
    return out(df, result, join, dropna)


def ROC(df, columns, n, join=None, dropna=True):
    """
    Rate of Change
    """
    result = sel_columns(df, columns, join)
    M = result.diff(n)
    N = result.shift(n)
    result = M / N * 100
    return out(df, result, join, dropna)


def ROCP(df, columns, n, join=None, dropna=True):
    """
    Rate of change Percentage 
    """
    result = sel_columns(df, columns, join)
    M = result.diff(n)
    N = result.shift(n)
    result = M / N
    return out(df, result, join, dropna)


def ATR(df, n, high_column='High', low_column='Low', close_column='Close', join=None, dropna=True):
    """
    Average True Range
    """
    high_series = df[high_column]
    low_series = df[low_column]
    close_prev_series = df[close_column].shift(1)
    tr = np.max((
        (high_series.values - low_series.values),
        np.abs(high_series.values - close_prev_series.values),
        np.abs(low_series.values - close_prev_series.values),
    ), 0)

    tr = pd.Series(tr, name=type(join) is list and join[0] or join)
    if len(tr) > n:
        tr[n] = tr[1:n+1].mean()
        nm1 = n - 1
        for i in range(n+1, len(tr)):
            tr[i] = (tr[i-1] * nm1 + tr[i]) / n

    tr[:n] = np.nan
    return out(df, tr, join, dropna)


def BBANDS(df, columns, n, join=None, dropna=True):
    """
    Bollinger Bands
    Example:  
        BBANDS(df, ['Close', 'VWAP'], 20, join=[
            ['Close_BBUp', 'Close_MA20', 'Close_BBDown'], 
            ['VWAP_BBUp', 'VWAP_MA20', 'VWAP_BBDown'], 
        ])
    """
    ma = MA(df, columns, n, dropna=dropna)
    std = STDDEV(df, columns, n, dropna=dropna)
    b1 = ma + 2 * std
    b2 = ma - 2 * std

    result = join_result([b1, ma, b2], columns, join)
    return out(df, result, join, dropna)


def MACD(df, columns, n_fast, n_slow, n_signal, join=None, dropna=True):
    """
    MACD, MACD Signal and MACD difference
    Example:  
        MACD(df, ['Close', 'VWAP'], 12, 26, 9, join=[
            ['Close_MACD', 'Close_MACDSIGN', 'Close_MACDHIST'], 
            ['VWAP_MACD', 'VWAP_MACDSIGN'', 'VWAP_MACDHIST'], 
        ])
    """
    assert(n_slow > n_fast)
    fast = EMA(df.iloc[n_slow-n_fast:], columns, n_fast, dropna=dropna, min_periods=n_fast)
    slow = EMA(df, columns, n_slow, dropna=dropna, min_periods=n_slow)
    macd = fast - slow

    # first drop nan, for calc mean on first row
    sign = EMA(macd[n_slow-1:], columns, n_signal, dropna=dropna, min_periods=n_signal)
    # and then restore nan row, for row count unchanged
    sign = sign.reindex_like(macd)

    macd.iloc[:n_slow + n_signal-2] = np.nan
    hist = macd - sign

    result = join_result([macd, sign, hist], columns, join)
    return out(df, result, join, dropna)


def RSI(df, columns, n, join=None, dropna=True):
    """
    Relative Strength Index
    """
    sel_df = sel_columns(df, columns, None)
    change = sel_df.diff(1)

    up = change.clip_lower(0)
    up.iloc[:n] = np.mean(up.values[1:n])
    up = up.ewm(com=n-1, adjust=False).mean()

    down = -change.clip_upper(0)
    down.iloc[:n] = np.mean(down.values[1:n])
    down = down.ewm(com=n-1, adjust=False).mean()

    # result = np.where(down == 0, 100, np.where(up == 0, 0, rsi))
    rsi = 100 - (100 / (1 + up / down))
    rsi[down == 0] = 100
    rsi[(down != 0) & (up == 0)] = 0
    rsi.iloc[:n] = np.nan
    result = rsi
    return out(df, result, join, dropna)


