
** Deprecated, please use spectre: https://github.com/Heerozh/spectre **

pandas\_talib
=============

纯Pandas实现的指标库，用于技术分析（目前只有常用指标）。

 - 兼容pandas 0.19
 - 跑通了所有ta-lib的对比测试，确保计算结果的正确。


----------------------



A Python Pandas implementation of technical indicators (Currently only common indicators)

 - Compatible with pandas 0.19
 - Pass all comparison test with the TA-Lib to ensure that the results are correct.


Usage
~~~~~~~~~~~~~~
Copy pandas_talib folder to your project path.

::

   import pandas as pd
   from pandas_talib import *

   df = pd.read_csv('./data/AAPL.csv')
   df = EMA(df, ['Close', 'Volume'], n=20, join=['Close_EMA20', 'Volume_EMA20'])
   df.tail()

Output::

                Date    Open    High     Low   Close      Volume  Ex-Dividend  \
    9149  2017-03-27  139.39  141.22  138.62  140.88  23374772.0          0.0
    9150  2017-03-28  140.91  144.04  140.62  143.80  33193535.0          0.0
    9151  2017-03-29  143.68  144.49  143.19  144.12  28979571.0          0.0
    9152  2017-03-30  144.19  144.50  143.50  143.93  21064727.0          0.0
    9153  2017-03-31  143.72  144.27  143.01  143.71  19228867.0          0.0
          Split Ratio  Adj. Open  Adj. High  Adj. Low  Adj. Close  Adj. Volume  \
    9149          1.0     139.39     141.22    138.62      140.88   23374772.0
    9150          1.0     140.91     144.04    140.62      143.80   33193535.0
    9151          1.0     143.68     144.49    143.19      144.12   28979571.0
    9152          1.0     144.19     144.50    143.50      143.93   21064727.0
    9153          1.0     143.72     144.27    143.01      143.71   19228867.0
          Close_EMA20  Volume_EMA20
    9149   139.153709  2.483257e+07
    9150   139.596213  2.562885e+07
    9151   140.027050  2.594797e+07
    9152   140.398759  2.548290e+07
    9153   140.714115  2.488728e+07





Run unit tests
~~~~~~~~~~~~~~

Run all unit tests

::

    $ nosetests -s -v


