"""
ta-lib tested pure pandas implement, 
Originated from https://github.com/femtotrader/pandas_talib

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



#
# def PPSR(df):
#     """
#     Pivot Points, Supports and Resistances
#     """
#     PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)
#     R1 = pd.Series(2 * PP - df['Low'])
#     S1 = pd.Series(2 * PP - df['High'])
#     R2 = pd.Series(PP + df['High'] - df['Low'])
#     S2 = pd.Series(PP - df['High'] + df['Low'])
#     R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))
#     S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))
#     result = pd.DataFrame([PP, R1, S1, R2, S2, R3, S3]).transpose()
#     return out(df, result, join, dropna)
#
#
# def STOK(df):
#     """
#     Stochastic oscillator %K
#     """
#     result = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']),
#                        name='SO%k')
#     return out(df, result, join, dropna)
#
#
# def STO(df, n):
#     """
#     Stochastic oscillator %D
#     """
#     SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']),
#                     name='SO%k')
#     result = pd.Series(pd.ewma(SOk, span=n, min_periods=n - 1),
#                        name='SO%d_' + str(n))
#     return out(df, result, join, dropna)
#
#
# def TRIX(df, n):
#     """
#     Trix
#     """
#     EX1 = pd.ewma(df['Close'], span=n, min_periods=n - 1)
#     EX2 = pd.ewma(EX1, span=n, min_periods=n - 1)
#     EX3 = pd.ewma(EX2, span=n, min_periods=n - 1)
#     i = 0
#     ROC_l = [0]
#     while i + 1 <= len(df) - 1:  # df.index[-1]:
#         ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
#         ROC_l.append(ROC)
#         i = i + 1
#     result = pd.Series(ROC_l, name='Trix_' + str(n))
#     return out(df, result, join, dropna)
#
#
# def ADX(df, n, n_ADX):
#     """
#     Average Directional Movement Index
#     """
#     i = 0
#     UpI = []
#     DoI = []
#     while i + 1 <= len(df) - 1:  # df.index[-1]:
#         UpMove = df.get_value(i + 1, 'High') - df.get_value(i, 'High')
#         DoMove = df.get_value(i, 'Low') - df.get_value(i + 1, 'Low')
#         if UpMove > DoMove and UpMove > 0:
#             UpD = UpMove
#         else:
#             UpD = 0
#         UpI.append(UpD)
#         if DoMove > UpMove and DoMove > 0:
#             DoD = DoMove
#         else:
#             DoD = 0
#         DoI.append(DoD)
#         i = i + 1
#     i = 0
#     TR_l = [0]
#     while i < len(df) - 1:  # df.index[-1]:
#         TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(
#             df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
#         TR_l.append(TR)
#         i = i + 1
#     TR_s = pd.Series(TR_l)
#     ATR = pd.Series(pd.ewma(TR_s, span=n, min_periods=n))
#     UpI = pd.Series(UpI)
#     DoI = pd.Series(DoI)
#     PosDI = pd.Series(pd.ewma(UpI, span=n, min_periods=n - 1) / ATR)
#     NegDI = pd.Series(pd.ewma(DoI, span=n, min_periods=n - 1) / ATR)
#     result = pd.Series(pd.ewma(abs(PosDI - NegDI) / (PosDI + NegDI), span=n_ADX,
#                                min_periods=n_ADX - 1),
#                        name='ADX_' + str(n) + '_' + str(n_ADX))
#     return out(df, result, join, dropna)
#
#
# def MassI(df):
#     """
#     Mass Index
#     """
#     Range = df['High'] - df['Low']
#     EX1 = pd.ewma(Range, span=9, min_periods=8)
#     EX2 = pd.ewma(EX1, span=9, min_periods=8)
#     Mass = EX1 / EX2
#     result = pd.Series(pd.rolling_sum(Mass, 25), name='Mass Index')
#     return out(df, result, join, dropna)
#
#
# def Vortex(df, n):
#     """
#     Vortex Indicator
#     """
#     i = 0
#     TR = [0]
#     while i < len(df) - 1:  # df.index[-1]:
#         Range = max(df.get_value(i + 1, 'High'),
#                     df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'),
#                                                     df.get_value(i, 'Close'))
#         TR.append(Range)
#         i = i + 1
#     i = 0
#     VM = [0]
#     while i < len(df) - 1:  # df.index[-1]:
#         Range = abs(df.get_value(i + 1, 'High') - df.get_value(i, 'Low')) - abs(
#             df.get_value(i + 1, 'Low') - df.get_value(i, 'High'))
#         VM.append(Range)
#         i = i + 1
#     result = pd.Series(
#         pd.rolling_sum(pd.Series(VM), n) / pd.rolling_sum(pd.Series(TR), n),
#         name='Vortex_' + str(n))
#     return out(df, result, join, dropna)
#
#
# def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
#     """
#     KST Oscillator
#     """
#     M = df['Close'].diff(r1 - 1)
#     N = df['Close'].shift(r1 - 1)
#     ROC1 = M / N
#     M = df['Close'].diff(r2 - 1)
#     N = df['Close'].shift(r2 - 1)
#     ROC2 = M / N
#     M = df['Close'].diff(r3 - 1)
#     N = df['Close'].shift(r3 - 1)
#     ROC3 = M / N
#     M = df['Close'].diff(r4 - 1)
#     N = df['Close'].shift(r4 - 1)
#     ROC4 = M / N
#     result = pd.Series(pd.rolling_sum(ROC1, n1) + pd.rolling_sum(ROC2,
#                                                                  n2) * 2 + pd.rolling_sum(
#         ROC3, n3) * 3 + pd.rolling_sum(ROC4, n4) * 4,
#                        name='KST_' + str(r1) + '_' + str(r2) + '_' + str(
#                            r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(
#                            n2) + '_' + str(n3) + '_' + str(n4))
#     return out(df, result, join, dropna)
#
#
# def TSI(df, r, s):
#     """
#     True Strength Index
#     """
#     M = pd.Series(df['Close'].diff(1))
#     aM = abs(M)
#     EMA1 = pd.Series(pd.ewma(M, span=r, min_periods=r - 1))
#     aEMA1 = pd.Series(pd.ewma(aM, span=r, min_periods=r - 1))
#     EMA2 = pd.Series(pd.ewma(EMA1, span=s, min_periods=s - 1))
#     aEMA2 = pd.Series(pd.ewma(aEMA1, span=s, min_periods=s - 1))
#     result = pd.Series(EMA2 / aEMA2, name='TSI_' + str(r) + '_' + str(s))
#     return out(df, result, join, dropna)
#
#
# def ACCDIST(df, n):
#     """
#     Accumulation/Distribution
#     """
#     ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * \
#          df['Volume']
#     M = ad.diff(n - 1)
#     N = ad.shift(n - 1)
#     ROC = M / N
#     result = pd.Series(ROC, name='Acc/Dist_ROC_' + str(n))
#     return out(df, result, join, dropna)
#
#
# def Chaikin(df):
#     """
#     Chaikin Oscillator
#     """
#     ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * \
#          df['Volume']
#     result = pd.Series(pd.ewma(ad, span=3, min_periods=2) - pd.ewma(ad, span=10,
#                                                                     min_periods=9),
#                        name='Chaikin')
#     return out(df, result, join, dropna)
#
#
# def MFI(df, n):
#     """
#     Money Flow Index and Ratio
#     """
#     PP = (df['High'] + df['Low'] + df['Close']) / 3
#     i = 0
#     PosMF = [0]
#     while i < len(df) - 1:  # df.index[-1]:
#         if PP[i + 1] > PP[i]:
#             PosMF.append(PP[i + 1] * df.get_value(i + 1, 'Volume'))
#         else:
#             PosMF.append(0)
#         i = i + 1
#     PosMF = pd.Series(PosMF)
#     TotMF = PP * df['Volume']
#     MFR = pd.Series(PosMF / TotMF)
#     result = pd.Series(pd.rolling_mean(MFR, n), name='MFI_' + str(n))
#     return out(df, result, join, dropna)
#
#
# def OBV(df, n):
#     """
#     On-balance Volume
#     """
#     i = 0
#     OBV = [0]
#     while i < len(df) - 1:  # df.index[-1]:
#         if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') > 0:
#             OBV.append(df.get_value(i + 1, 'Volume'))
#         if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') == 0:
#             OBV.append(0)
#         if df.get_value(i + 1, 'Close') - df.get_value(i, 'Close') < 0:
#             OBV.append(-df.get_value(i + 1, 'Volume'))
#         i = i + 1
#     OBV = pd.Series(OBV)
#     result = pd.Series(pd.rolling_mean(OBV, n), name='OBV_' + str(n))
#     return out(df, result, join, dropna)
#
#
# def FORCE(df, n):
#     """
#     Force Index
#     """
#     result = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n),
#                        name='Force_' + str(n))
#     return out(df, result, join, dropna)
#
#
# def EOM(df, n):
#     """
#     Ease of Movement
#     """
#     EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (
#     df['High'] - df['Low']) / (2 * df['Volume'])
#     result = pd.Series(pd.rolling_mean(EoM, n), name='EoM_' + str(n))
#     return out(df, result, join, dropna)
#
#
# def CCI(df, n):
#     """
#     Commodity Channel Index
#     """
#     PP = (df['High'] + df['Low'] + df['Close']) / 3
#     result = pd.Series((PP - pd.rolling_mean(PP, n)) / pd.rolling_std(PP, n),
#                        name='CCI_' + str(n))
#     return out(df, result, join, dropna)
#
#
# def COPP(df, n):
#     """
#     Coppock Curve
#     """
#     M = df['Close'].diff(int(n * 11 / 10) - 1)
#     N = df['Close'].shift(int(n * 11 / 10) - 1)
#     ROC1 = M / N
#     M = df['Close'].diff(int(n * 14 / 10) - 1)
#     N = df['Close'].shift(int(n * 14 / 10) - 1)
#     ROC2 = M / N
#     result = pd.Series(pd.ewma(ROC1 + ROC2, span=n, min_periods=n),
#                        name='Copp_' + str(n))
#     return out(df, result, join, dropna)
#
#
# def KELCH(df, n):
#     """
#     Keltner Channel
#     """
#     KelChM = pd.Series(
#         pd.rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n),
#         name='KelChM_' + str(n))
#     KelChU = pd.Series(
#         pd.rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n),
#         name='KelChU_' + str(n))
#     KelChD = pd.Series(
#         pd.rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n),
#         name='KelChD_' + str(n))
#     result = pd.DataFrame([KelChM, KelChU, KelChD]).transpose()
#     return out(df, result, join, dropna)
#
#
# def ULTOSC(df):
#     """
#     Ultimate Oscillator
#     """
#     i = 0
#     TR_l = [0]
#     BP_l = [0]
#     while i < len(df) - 1:  # df.index[-1]:
#         TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(
#             df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))
#         TR_l.append(TR)
#         BP = df.get_value(i + 1, 'Close') - min(df.get_value(i + 1, 'Low'),
#                                                 df.get_value(i, 'Close'))
#         BP_l.append(BP)
#         i = i + 1
#     result = pd.Series((4 * pd.rolling_sum(pd.Series(BP_l), 7) / pd.rolling_sum(
#         pd.Series(TR_l), 7)) + (
#                        2 * pd.rolling_sum(pd.Series(BP_l), 14) / pd.rolling_sum(
#                            pd.Series(TR_l), 14)) + (
#                        pd.rolling_sum(pd.Series(BP_l), 28) / pd.rolling_sum(
#                            pd.Series(TR_l), 28)), name='Ultimate_Osc')
#     return out(df, result, join, dropna)
#
#
# def DONCH(df, n):
#     """
#     Donchian Channel
#     """
#     i = 0
#     DC_l = []
#     while i < n - 1:
#         DC_l.append(0)
#         i = i + 1
#     i = 0
#     while i + n - 1 < len(df) - 1:  # df.index[-1]:
#         DC = max(df['High'].ix[i:i + n - 1]) - min(df['Low'].ix[i:i + n - 1])
#         DC_l.append(DC)
#         i = i + 1
#     DonCh = pd.Series(DC_l, name='Donchian_' + str(n))
#     result = DonCh.shift(n - 1)
#     return out(df, result, join, dropna)
#


