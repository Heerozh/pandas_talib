"""
ta-lib tested pure pandas implement

Repository:
https://github.com/Heerozh/pandas_talib
"""
import pandas as pd
import numpy as np

__all__ = ['MA', 'SMA', 'EMA', 'STDDEV', 'MOM', 'ROC', 'ROCP', 'ATR',
           'BBANDS', 'MACD', 'RSI', 'MAX', 'MIN']


def out(df, result, join, dropna, dtype):

    if dtype:
        if isinstance(dtype, (list, tuple)):
            for i, v in enumerate(dtype):
                result.iloc[:, i] = result.iloc[:, i].astype(v)
                print(i, v, result.info())
        else:
            result = result.astype(dtype)

    if join:
        result = df.join(result, how='inner')

    if dropna:
        result.dropna(inplace=True)

    return result


def rename_columns(df, columns, new_names):
    assert len(columns) == len(new_names), \
        '"join" length needs to be same as "columns".'
    return df.rename(columns=dict(zip(columns, new_names)))


def sel_columns(df, columns, new_names):
    if type(columns) is pd.Series:
        result = columns
        columns = result.name
    elif type(columns) is pd.DataFrame:
        result = columns
        columns = result.columns.values
    else:
        if type(columns) is str:
            columns = [columns]

        if type(df) is pd.Series:
            result = df
        else:
            result = df[columns]

    if new_names:
        result = rename_columns(result, columns, new_names)

    return result


def join_result(dfs, columns, new_names):
    if type(columns) is str:
        columns = [columns]

    if new_names is None or new_names is False:
        join = np.array([[s + str(i) for i in range(len(dfs))]
                         for s in columns])
    else:
        join = np.array(new_names)

    result = dfs[0]
    for i in range(len(dfs)):
        dfs[i].rename(columns=dict(zip(columns, join[:, i])), inplace=True)
        if i > 0:
            result = result.join(dfs[i], how='inner')
    return result


def MA(df, columns, n, join=None, dropna=False, dtype=np.float32):
    """
    Moving Average
    """
    result = sel_columns(df, columns, join)
    result = result.rolling(n).mean()
    return out(df, result, bool(join), dropna, dtype)


SMA = MA


def EMA(df, columns, n, join=None, dropna=False, min_periods=-1,
        dtype=np.float32):
    """
    Exponential Moving Average
    """
    result = sel_columns(df, columns, join).copy()

    ma = np.mean(result.values[:n], axis=0)
    result.iloc[:n] = np.resize(ma, (n, len(ma)))
    if min_periods == -1:
        min_periods = n

    # print(n, result.head())
    result = result.ewm(span=n, min_periods=min_periods, adjust=False).mean()
    return out(df, result, bool(join), dropna, dtype)


def STDDEV(df, columns, n, join=None, dropna=False, dtype=None):
    """
    Standard Deviation
    """
    result = sel_columns(df, columns, join)
    result = result.rolling(n).std(ddof=0)
    return out(df, result, bool(join), dropna, dtype)


def MOM(df, columns, n, join=None, dropna=False, dtype=None):
    """
    Momentum
    """
    result = sel_columns(df, columns, join)
    result = result.diff(n)
    return out(df, result, bool(join), dropna, dtype)


def ROC(df, columns, n, join=None, dropna=False, dtype=np.float32):
    """
    Rate of Change
    """
    result = sel_columns(df, columns, join)
    M = result.diff(n)
    N = result.shift(n)
    result = M / N * 100
    return out(df, result, bool(join), dropna, dtype)


def ROCP(df, columns, n, join=None, dropna=False, dtype=np.float32):
    """
    Rate of change Percentage 
    """
    result = sel_columns(df, columns, join)
    M = result.diff(n)
    N = result.shift(n)
    result = M / N
    return out(df, result, bool(join), dropna, dtype)


def ATR(df, n, high_column='High', low_column='Low', close_column='Close',
        join=None, dropna=False, dtype=None):
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
    return out(df, tr, bool(join), dropna, dtype)


def BBANDS(df, columns, n, join=None, dropna=False, normalize=False,
           dtype=np.float32):
    """
    Bollinger Bands
    Example:  
        BBANDS(df, ['Close', 'VWAP'], 20, join=[
            ['Close_BBUp', 'Close_MA20', 'Close_BBDown'], 
            ['VWAP_BBUp', 'VWAP_MA20', 'VWAP_BBDown'], 
        ])
    :param normalize: Normalized Bollinger Bands, range -1.0 to 1.0
    """
    ma = MA(df, columns, n, dropna=dropna, dtype=np.float64)
    std = STDDEV(df, columns, n, dropna=dropna, dtype=np.float64)
    if not normalize:
        b1 = ma + 2 * std
        b2 = ma - 2 * std
        result = join_result([b1, ma, b2], columns, join)
    else:
        nbb = sel_columns(df, columns, new_names=None)
        nbb = (nbb - ma) / (2 * std)
        result = rename_columns(nbb, columns, join)
    return out(df, result, bool(join), dropna, dtype)


def MACD(df, columns, n_fast, n_slow, n_signal, join=None, dropna=False,
         normalize=False, dtype=np.float32):
    """
    MACD, MACD Signal and MACD difference
    Example:  
        MACD(df, ['Close', 'VWAP'], 12, 26, 9, join=[
            ['Close_MACD', 'Close_MACDSIGN', 'Close_MACDHIST'], 
            ['VWAP_MACD', 'VWAP_MACDSIGN'', 'VWAP_MACDHIST'], 
        ])
    """
    assert n_slow > n_fast, '"n_slow" needs to be greater than "n_fast"'
    fast = EMA(df.iloc[n_slow-n_fast:], columns, n_fast, dropna=dropna,
               min_periods=n_fast, dtype=np.float64)
    slow = EMA(df, columns, n_slow, dropna=dropna,
               min_periods=n_slow, dtype=np.float64)
    macd = fast - slow

    # first drop nan, for calc mean on first row
    sign = EMA(macd[n_slow-1:], columns, n_signal, dropna=dropna,
               min_periods=n_signal, dtype=np.float64)
    # and then restore nan row, for row count unchanged
    sign = sign.reindex_like(macd)

    macd.iloc[:n_slow + n_signal-2] = np.nan
    hist = macd - sign

    if not normalize:
        assert not join or (len(join), len(join[0])) == (len(columns), 3), \
            'join: shape must be {}'.format((len(columns), 3))
        result = join_result([macd, sign, hist], columns, join)
    else:
        assert not join or len(join) == len(columns) and \
            type(join[0]) is not (list or tuple), \
            'join: shape must be ({})'.format(len(columns))
        result = rename_columns(hist, columns, join)

    return out(df, result, bool(join), dropna, dtype)


def RSI(df, columns, n, join=None, dropna=False, normalize=False,
        dtype=np.float32):
    """
    Relative Strength Index
    :param normalize: Normalized RSI, range -1.0 to 1.0
    """
    sel_df = sel_columns(df, columns, join)
    change = sel_df.diff(1)

    up = change.clip_lower(0)
    ma = np.mean(up.values[1:n], axis=0)
    up.iloc[:n] = np.resize(ma, (n, len(ma)))
    up = up.ewm(com=n-1, adjust=False).mean()

    down = -change.clip_upper(0)
    ma = np.mean(down.values[1:n], axis=0)
    down.iloc[:n] = np.resize(ma, (n, len(ma)))
    down = down.ewm(com=n-1, adjust=False).mean()

    # result = np.where(down == 0, 100, np.where(up == 0, 0, rsi))
    if not normalize:
        rsi = 100 - (100 / (1 + up / down))
        rsi[down == 0] = 100
        rsi[(down != 0) & (up == 0)] = 0
    else:
        rsi = 1 - (2 / (1 + up / down))
        rsi[down == 0] = 1
        rsi[(down != 0) & (up == 0)] = -1

    rsi.iloc[:n] = np.nan
    result = rsi
    return out(df, result, bool(join), dropna, dtype)


def MAX(df, columns, n, join=None, dropna=False, dtype=np.float32):
    """
    Highest value over a specified period
    """
    result = sel_columns(df, columns, join)
    result = result.rolling(n).max()
    return out(df, result, bool(join), dropna, dtype)


def MIN(df, columns, n, join=None, dropna=False, dtype=np.float32):
    """
    Lowest value over a specified period
    """
    result = sel_columns(df, columns, join)
    result = result.rolling(n).min()
    return out(df, result, bool(join), dropna, dtype)

