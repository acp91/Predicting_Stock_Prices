import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import yahoo_fin
import datetime
import yahoo_fin
import yahoo_fin.stock_info as yfsi
import seaborn as sns
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

    data_sp500, data_dow, data_ftse100, data_ftse250, data_nasdaq, data_nifty51 = 0, 0, 0, 0, 0, 0

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

    return data_sp500, data_dow, data_ftse100, data_ftse250, data_nasdaq, data_nifty50

def recommend(data_index, compare_value=1,  compare_type='below', sector=['Energy'], offset_start=-1, offset_end=-10,
              start='2021-01-01', end=(datetime.date.today()+datetime.timedelta(1)).strftime("%Y-%m-%d"), type='Close'):
    '''

    :param data_index: index data retrieved with get_index_data() function
    :param compare_value: value to compare rise/drop in price against for recommendation
    :param compare_type: either 'below' or 'above'
    :param sector: industry sector. check self.sectors for list of valid inputs
    :param start: start data for analysis/plot
    :param end: end day for analysis/plot
    :param type: type of data to take, 'Open', 'High', 'Low', 'Close'
    :return:
    '''

    # set new dict for recommended tickers
    recommended_tickers = dict()
    # keep track of recommended tickers and trends
    trends = dict()

    # average prices trend
    sector_trend = 0

    if sector != '':
        data_index = data_index[data_index['Sector'].isin(sector)]

    sector_size = data_index.shape[0]
    print('Total number of tickers to check is {}'.format(sector_size))

    for i, ticker in enumerate(data_index['Symbol']):

        with HiddenPrints():
            # get price history based on type input (default Close)
            ticker_df = yf.download(ticker, start=start, end=end)[type]

        # normalized price set
        ticker_df_normalize = (ticker_df - ticker_df[0]) / ticker_df[0]

        # save it into 'sector_tren
        sector_trend += ticker_df_normalize

        # check if enough historical data available
        if len(ticker_df) > 0:
            current_ticker_last = ticker_df.iloc[offset_start]
            current_ticker_previous = ticker_df.iloc[offset_end]
            current_ticker_yesterday = ticker_df.iloc[offset_start-1]
            compare_calc = current_ticker_last / current_ticker_previous

            if (compare_calc < compare_value and compare_type=='below') or (compare_calc > compare_value and compare_type=='above'):
                my_ticker_info = yf.Ticker(ticker).info
                price_start = ticker_df.iloc[0]
                ticker_median_price = my_ticker_info['targetMedianPrice']
                if ticker_median_price != None:
                    upside = '{:.0%}'.format((ticker_median_price - current_ticker_last) / current_ticker_last)
                else:
                    upside = np.nan

                print('Ticker {} is part of the recommendation. Still {} tickers to check'.format(ticker, sector_size-i))
                # add data to dictionary
                recommended_tickers[ticker] = [current_ticker_last,
                                           current_ticker_previous,
                                           compare_calc, ticker_median_price,
                                           upside,
                                           my_ticker_info['recommendationKey'],
                                           my_ticker_info['numberOfAnalystOpinions'],
                                           '{:.0%}'.format((current_ticker_last - price_start) / price_start),
                                           '{:.0%}'.format((current_ticker_last - current_ticker_yesterday) / current_ticker_yesterday)]

                # add to trends
                trends[ticker + ' - ' + data_index[data_index['Symbol'] == ticker].Security.tolist()[0]] = ticker_df_normalize

    # store data frame for recommended tickers
    try:
        recommended_tickers = pd.DataFrame(recommended_tickers).transpose()
        recommended_tickers.columns = ['Last', 'Previous', 'Compare', 'Target', 'Upside', 'Recommendation', '# of Analysts', 'YTD', 'Daily Change']
    except:
        pass

    # store trends of recommended tickers
    trends = pd.DataFrame(trends)

    # store overall industry trend
    trends['Overall Sector'] = sector_trend / len(data_index)

    return recommended_tickers, trends

def plot_graphs(df, x_size=12, y_size=8, font_size='8', style='fivethirtyeight', style_sns='white'):

    '''

    :param x_size: size of x axis
    :param y_size: size ox y axis
    :return: none - plots graph
    '''

    # get dates and format them
    date_ticks = df.reset_index()['Date'].tolist()

    # set figure size for the plot
    plt.figure(figsize=(x_size, y_size))
    plt.rcParams['font.size'] = font_size

    if style != '':
        plt.style.use(style)
    if style_sns != '':
        sns.set_style(style_sns)

    # plot all separate tickets
    for column in df.columns:
        if column == 'Overall Sector':
            plt.plot(df[column], label=column, linestyle='-', linewidth=3, color='purple')
        else:
            plt.plot(df[column], label=column, linestyle='--', linewidth=2)

    for column in (df.columns):
        plt.annotate('%0.2f' % df[column].iloc[-1], xy=(1, df[column].iloc[-1]), xytext=(8, 0),
                     xycoords=('axes fraction', 'data'), textcoords='offset points')

    plt.grid(visible=True)
    plt.legend(loc='upper left') # prop={'size': 10})
    plt.xticks(date_ticks[1::15], rotation='20')
    plt.xlabel('Dates')
    plt.ylabel('Return (relative)')

def top_bottom(df, sector=['Energy']):
    '''

    :param sector: list of sectors to produce top_bottom view
    :return: none - plots graph
    '''

    if sector != '':
        df = df[df['Sector'].isin(sector)]

    top_bottom_df = dict()
    top_bottom__sector = list()

    for ticker in df['Symbol']:
        ticker_prices = yf.Ticker(ticker).history(period='2d')['Close']
        if len(ticker_prices) > 0:
            price_today = ticker_prices[-1]
            price_yesterday = ticker_prices[0]
            daily_return = '{:.2%}'.format((price_today - price_yesterday) / price_yesterday)
            top_bottom_df[ticker] = [price_today, price_yesterday, daily_return]
            if sum(df.columns.isin(['Sector'])==True) != 0:
                top_bottom__sector.append(df[df['Symbol']==ticker]['Sector'].iloc[0])

    top_bottom_df = pd.DataFrame(top_bottom_df).transpose().reset_index()
    top_bottom_df.columns = ['Symbol', 'Today', 'Yesterday', 'Daily Return']
    if sum(df.columns.isin(['Sector'])==True) != 0:
        top_bottom_df['Sector'] = top_bottom__sector
    top_bottom_df['New'] = top_bottom_df['Daily Return'].str.strip('%').astype(float)
    top_bottom_df['New'].fillna(0, inplace=True)
    top_bottom_df.sort_values('New', ascending=False, inplace=True, ignore_index=True)
    top_bottom_df.drop(['New'], axis=1, inplace=True)

    return top_bottom_df