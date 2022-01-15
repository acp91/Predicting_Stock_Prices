import pandas as pd
import yahoo_fin.stock_info as yfsi
import os, sys

pd.set_option('display.max_columns',100)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def load_index_data():
    '''

    :param
    :return: return 6 data frames in this order:
        data_sp500, data_dow, data_ftse100, data_ftse250, data_nasdaq, data_nifty50

        data frame as standardize:
            1. where Sector available, rename it everywhere to Sector
            2. Symbol renamed everywhere to Symbol
    '''

    data_sp500, data_dow, data_ftse100, data_ftse250, data_nasdaq = 0, 0, 0, 0, 0

    # SP500
    try:
        data_sp500 = yfsi.tickers_sp500(True)
        data_sp500 = data_sp500.rename(columns={'GICS Sector': 'Sector'})
    except:
        print('Data cant be retrieved for tickers_sp500')
        pass

    # DJIA
    try:
        data_dow = yfsi.tickers_dow(True)
        data_dow = data_dow.rename(columns={'Industry': 'Sector', 'Company': 'Security'})
    except:
        print('Data cant be retrieved for data_dow')
        pass

    # FTSE100
    try:
        data_ftse100 = yfsi.tickers_ftse100(True)
        data_ftse100 = data_ftse100.rename(\
                                           columns={'FTSE Industry Classification Benchmark sector[13]': 'Sector', 'EPIC': 'Symbol', 'Company': 'Security'})
    except:
        print('Data cant be retrieved for tickers_ftse100')
        pass

    # FTSE250
    try:
        data_ftse250 = yfsi.tickers_ftse250(True)
        data_ftse250 = data_ftse250.rename(columns={'Ticker': 'Symbol', 'Company': 'Security'})
    except:
        print('Data cant be retrieved for tickers_ftse250')
        pass

    # NASDAQ
    try:
        data_nasdaq = yfsi.tickers_nasdaq(True)
        data_nasdaq = data_nasdaq.rename(columns={'Security Name': 'Security'})
    except:
        print('Data cant be retrieved for tickers_nasdaq')
        pass

    # NIFTY50
    try:
        data_nifty50 = yfsi.tickers_nifty50(True)
        data_nifty50 = data_nifty50.rename(columns={'Company Name': 'Security'})
    except:
        print('Data cant be retrieved for tickers_nifty50')
        pass

    return data_sp500, data_dow, data_ftse100, data_ftse250, data_nasdaq