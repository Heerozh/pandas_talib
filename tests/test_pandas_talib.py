#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ta-lib source code see:
# https://sourceforge.net/p/ta-lib/code/HEAD/tree/trunk/ta-lib/c/src/ta_func/ta_MA.c

import unittest

import pandas as pd
import numpy as np
import math
import talib
from pandas_talib import *

try:
    df = pd.read_csv('../data/AAPL.csv')
except OSError:
    import quandl
    df = quandl.get("WIKI/AAPL")
    df.to_csv('../data/AAPL.csv')


class TestFunctions(unittest.TestCase):

    def test_indicator_SMA(self):
        timeperiod = 10
        random_serie = pd.DataFrame(np.random.uniform(0, 1, size=10), columns=['last'])
        result = SMA(random_serie, 'last', timeperiod, join=False, dropna=False, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.SMA(random_serie['last'].values, timeperiod=10)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_MA(self):
        n = 5
        result = MA(df, 'Close', n, join=False, dropna=False, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.MA(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_EMA(self):
        n = 3
        result = EMA(df, 'Close', n, join=False, dropna=False, min_periods=n, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.EMA(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected[:])
        # test bug axis
        result = EMA(df, ['Close', 'Volume'], 20, join=['CloseEMA20', 'VolumeEMA20'], min_periods=-1)

    def test_indicator_MOM(self):
        n = 15
        result = MOM(df, 'Close', n, join=False, dropna=False, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.MOM(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_ROC(self):
        n = 1
        result = ROC(df, 'Close', n, join=False, dropna=False, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.ROC(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_ROCP(self):
        for n in [1, 5]:
            result = ROCP(df, 'Close', n, join=False, dropna=False, dtype=np.float64)
            isinstance(result, pd.DataFrame)
            expected = talib.ROCP(df['Close'].values, timeperiod=n)
            np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_ATR(self):
        n = 14
        result = ATR(df, n, join=False, dropna=False, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=n)
        # print('ATR', result.values[-10:])
        np.testing.assert_almost_equal(result.values, expected)

    def test_indicator_BBANDS(self):
        n = 20
        result = BBANDS(df, 'Close', n, join=False, dropna=False, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.BBANDS(df['Close'].values, timeperiod=n)
        # print('BBANDS', result.values[-10:])
        np.testing.assert_almost_equal(result.values.T, expected)

    def test_indicator_MACD(self):
        n_fast, n_slow, s_signal = 12, 26, 9
        result = MACD(df, 'Close', n_fast, n_slow, s_signal, join=False, dropna=False, dtype=np.float64)
        # print('MACDx', result.values.T[:, 20:50])
        isinstance(result, pd.DataFrame)
        expected = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        # print('MACDy', np.array(expected)[:, 20:50])
        np.testing.assert_almost_equal(result.values.T[:, :], np.array(expected)[:, :])

    def test_indicator_RSI(self):
        n = 14
        result = RSI(df, 'Close', n, join=False, dropna=False, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.RSI(df['Close'].values, timeperiod=n)
        # print('rsi x', result[0:50])
        # print('rsi y', pd.DataFrame(expected)[0:50])
        np.testing.assert_almost_equal(result.values[:, -1], expected[:])
        result = RSI(df, ['Close', 'Volume'], 3, join=['CloseRSI', 'VolumeRSI'])

    def test_indicator_STDDEV(self):
        n = 10
        result = STDDEV(df, 'Close', n, join=False, dropna=False, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.STDDEV(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

        rocp = ROCP(df, ['Close'], 3, dropna=True)
        result = STDDEV(df, rocp, n, join=False, dropna=False, dtype=np.float64)
        df2 = ROCP(df, ['Close'], 3, join=['Close_ROCP3'], dropna=True, dtype=np.float64)
        expected = talib.STDDEV(df2['Close_ROCP3'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_MAX(self):
        n = 5
        result = MAX(df, 'Close', n, join=False, dropna=False, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.MAX(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_MIN(self):
        n = 5
        result = MIN(df, 'Close', n, join=False, dropna=False, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.MIN(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_STOCH(self):
        n = 14
        result = STOCH(df, n, join=False, dropna=False, dtype=np.float64)
        isinstance(result, pd.DataFrame)
        expected = talib.STOCH(
            df['High'].values, df['Low'].values, df['Close'].values,
            fastk_period=n, slowk_period=1, slowd_period=3)
        print('kdj x', result.as_matrix()[0, 0:50])
        print('kdj y', np.array(expected)[0, 0:50])
        np.testing.assert_almost_equal(result.values, expected)
