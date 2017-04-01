\*\* Work in progress \*\*

pandas\_talib
=============

（未完成！）

纯Pandas实现的指标库，用于技术分析。

修改自 https://github.com/femtotrader/pandas_talib，
 - 兼容了pandas 0.19
 - 完善了接口使得方法可以更泛化，比如可以同时对多列操作。
 - 修改了错误，跑通了所有ta-lib的对比测试，确保计算结果的正确。


\*\* Work in progress \*\*

A Python Pandas implementation of technical indicators

Modified from https://github.com/femtotrader/pandas_talib,
 - compatible with pandas 0.19
 - Modified some interfaces makes the method more general, such as the multi-column support.
 - Pass all the comparison test of the original ta-lib to ensure that the results are correct.


Usage
~~~~~~~~~~~~~~
Copy pandas_talib folder to your project path.

::

   import pandas as pd
   from pandas_talib import *

   df = pd.read_csv('./data/AAPL_20140101_20141201.csv')
   df = EMA(df, n=20, columns=['Close', 'Volume'], join=['Close_MA20', 'Volume_MA20'])
   df.head()

Output::

             Date        Open        High         Low       Close       Volume  \
    0  2014-01-02  555.680008  557.029945  552.020004  553.129990   58671200.0
    1  2014-01-03  552.860023  553.699989  540.430046  540.980019   98116900.0
    2  2014-01-06  537.450005  546.800018  533.599983  543.930046  103152700.0
    3  2014-01-07  544.320015  545.960052  537.919975  540.039970   79302300.0
    4  2014-01-08  538.810036  545.559990  538.689980  543.459969   64632400.0
       Adj Close  Close_MA20   Volume_MA20
    0  76.765051  553.129990  5.867120e+07
    1  75.078841  551.972850  6.242793e+07
    2  75.488255  551.206869  6.630648e+07
    3  74.948378  550.143354  6.754418e+07
    4  75.423016  549.506842  6.726687e+07



Run unit tests
~~~~~~~~~~~~~~

Run all unit tests

::

    $ nosetests -s -v

Run a given test

::

    $ nosetests tests.test_pandas_talib:test_indicator_MA -s -v



