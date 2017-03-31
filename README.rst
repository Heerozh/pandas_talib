
\*\* Work in progress \*\*

pandas\_talib
=============

修改自 https://github.com/femtotrader/pandas_talib，兼容了pandas 0.19，以及修改了些接口比如可以同时对多列操作。

Modified from https://github.com/femtotrader/pandas_talib, now compatible with pandas 0.19,
and modified some of the interface, such as the multi-column support.

A Python Pandas implementation of technical indicators

Original version from:

-  `Bruno Franca <https://github.com/brunogfranca>`__

-  `panpanpandas <https://github.com/panpanpandas>`__

-  `Peter
   Bakker <https://www.quantopian.com/users/51d125a71144e60865000044>`__

See also:

-  `panpanpandas/ultrafinance <https://github.com/panpanpandas/ultrafinance>`__

-  `llazzaro/analyzer <https://github.com/llazzaro/analyzer>`__

-  https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code

If you are looking for a more complete set of technical indicators you
might have a look at this TA-Lib Python wrapper:
https://github.com/mrjbq7/ta-lib

Install
-------

A package is available and can be downloaded from PyPi and installed
using:

::

    $ pip install pandas_talib

Development
-----------

You can help to develop this library.

Issues
~~~~~~

You can submit issues using
https://github.com/femtotrader/pandas_talib/issues

Clone
~~~~~

You can clone repository to try to fix issues yourself using:

::

    $ git clone https://github.com/femtotrader/pandas_talib.git

Run unit tests
~~~~~~~~~~~~~~

Run all unit tests

::

    $ nosetests -s -v

Run a given test

::

    $ nosetests tests.test_pandas_talib:test_indicator_MA -s -v

Run samples
~~~~~~~~~~~

Run ``samples/main.py`` (from project root directory)

::

    $ python samples/main.py

Install development version
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    $ python setup.py install

or

::

    $ sudo pip install git+https://github.com/femtotrader/pandas_talib.git

Collaborating
~~~~~~~~~~~~~

-  Fork repository
-  Create a branch which fix a given issue
-  Submit pull requests

https://help.github.com/categories/collaborating/

