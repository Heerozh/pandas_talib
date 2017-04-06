#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import pandas as pd
import numpy as np
import math
import talib
from pandas_talib import *

try:
    df = pd.read_csv('./data/AAPL.csv')
except OSError:
    import quandl
    df = quandl.get("WIKI/AAPL")
    df.to_csv('./data/AAPL.csv')


class TestFunctions(unittest.TestCase):

    def test_indicator_SMA(self):
        timeperiod = 10
        random_serie = pd.DataFrame(np.random.uniform(0, 1, size=10), columns=['last'])
        result = SMA(random_serie, 'last', timeperiod, join=False, dropna=False)
        isinstance(result, pd.DataFrame)
        expected = talib.SMA(random_serie['last'].values, timeperiod=10)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_MA(self):
        n = 5
        result = MA(df, 'Close', n, join=False, dropna=False)
        isinstance(result, pd.DataFrame)
        expected = talib.MA(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_EMA(self):
        n = 3
        result = EMA(df, 'Close', n, join=False, dropna=False, min_periods=n)
        isinstance(result, pd.DataFrame)
        expected = talib.EMA(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected[:])

    def test_indicator_MOM(self):
        n = 15
        result = MOM(df, 'Close', n, join=False, dropna=False)
        isinstance(result, pd.DataFrame)
        expected = talib.MOM(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_ROC(self):
        n = 1
        result = ROC(df, 'Close', n, join=False, dropna=False)
        isinstance(result, pd.DataFrame)
        expected = talib.ROC(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_ROCP(self):
        n = 1
        result = ROCP(df, 'Close', n, join=False, dropna=False)
        isinstance(result, pd.DataFrame)
        expected = talib.ROCP(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)

    def test_indicator_ATR(self):
        n = 14
        result = ATR(df, n, join=False, dropna=False)
        isinstance(result, pd.DataFrame)
        expected = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=n)
        # print('ATR', result.values[-10:])
        np.testing.assert_almost_equal(result.values, expected)

    def test_indicator_BBANDS(self):
        n = 20
        result = BBANDS(df, 'Close', n, join=False, dropna=False)
        isinstance(result, pd.DataFrame)
        expected = talib.BBANDS(df['Close'].values, timeperiod=n)
        # print('BBANDS', result.values[-10:])
        np.testing.assert_almost_equal(result.values.T, expected)

    def test_indicator_MACD(self):
        n_fast, n_slow, s_signal = 12, 26, 9
        result = MACD(df, 'Close', n_fast, n_slow, s_signal, join=False, dropna=False)
        # print('MACDx', result.values.T[:, 20:50])
        isinstance(result, pd.DataFrame)
        expected = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        # print('MACDy', np.array(expected)[:, 20:50])
        np.testing.assert_almost_equal(result.values.T[:, :], np.array(expected)[:, :])

    def test_indicator_RSI(self):
        n = 14
        result = RSI(df, 'Close', n, join=False, dropna=False)
        isinstance(result, pd.DataFrame)
        expected = talib.RSI(df['Close'].values, timeperiod=n)
        # print('rsi x', result[0:50])
        # print('rsi y', pd.DataFrame(expected)[0:50])
        np.testing.assert_almost_equal(result.values[:, -1], expected[:])

    """
    def test_indicator_PPSR(self):
        result = PPSR(df)
        isinstance(result, pd.DataFrame)

    def test_indicator_STOK(self):
        result = STOK(df)
        isinstance(result, pd.DataFrame)

    def test_indicator_STO(self):
        n = 2
        result = STO(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_TRIX(self):
        n = 3
        result = TRIX(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_ADX(self):
        (n, n_ADX) = (2, 4)
        result = ADX(df, n, n_ADX)
        isinstance(result, pd.DataFrame)

    def test_indicator_MassI(self):
        result = MassI(df)
        isinstance(result, pd.DataFrame)

    def test_indicator_Vortex(self):
        n = 2
        result = Vortex(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_KST(self):
        (r1, r2, r3, r4, n1, n2, n3, n4) = (1, 2, 3, 4, 6, 7, 9, 9)
        result = KST(df, r1, r2, r3, r4, n1, n2, n3, n4)
        isinstance(result, pd.DataFrame)

    def test_indicator_TSI(self):
        (r, s) = (2, 4)
        result = TSI(df, r, s)
        isinstance(result, pd.DataFrame)

    def test_indicator_ACCDIST(self):
        n = 2
        result = ACCDIST(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_Chaikin(self):
        result = Chaikin(df)
        isinstance(result, pd.DataFrame)

    def test_indicator_MFI(self):
        n = 2
        result = MFI(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_OBV(self):
        n = 2
        result = OBV(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_FORCE(self):
        n = 2
        result = FORCE(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_EOM(self):
        n = 2
        result = EOM(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_CCI(self):
        n = 2
        result = CCI(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_COPP(self):
        n = 2
        result = COPP(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_COPP(self):
        n = 2
        result = COPP(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_KELCH(self):
        n = 2
        result = KELCH(df, n)
        isinstance(result, pd.DataFrame)

    def test_indicator_ULTOSC(self):
        n = 2
        result = ULTOSC(df)
        isinstance(result, pd.DataFrame)

    def test_indicator_DONCH(self):
        n = 2
        result = DONCH(df, n)
        isinstance(result, pd.DataFrame)
    """

    def test_indicator_STDDEV(self):
        n = 10
        result = STDDEV(df, 'Close', n, join=False, dropna=False)
        isinstance(result, pd.DataFrame)
        expected = talib.STDDEV(df['Close'].values, timeperiod=n)
        np.testing.assert_almost_equal(result.values[:, -1], expected)
