# overall path: http://127.0.0.1:5000/inputSymbol

from flask import Flask
from flask import render_template, request, jsonify
import joblib
from plotly.graph_objs import Bar
import plotly.express as px
import json
import plotly
import pandas as pd
from flask import Flask, render_template, request, flash

import pandas as pd
from pandas.tseries.offsets import BDay
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import yahoo_fin
import datetime
from datetime import date
import yahoo_fin
import yahoo_fin.stock_info as yfsi
import seaborn as sns
import os, sys
from pandas.tseries.offsets import BDay
import re

import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

from lib.finRecomm import load_index_data
from lib.contentBasedRecom import createNames, colsForMatrix, returnColumn, createDf, getRecommendations, FunkSVD
from lib.LSTM import LstmModel

# create a Flask for the app
app = Flask(__name__)
app.secret_key = "1"
#app = Flask(static_folder='C:\\Users\Andre\Desktop\Programming\StockRecommender\app\static')

@app.route("/inputSymbol")
def index():
    flash("What stock are you interested in?")
    return render_template('index.html')

@app.route("/recommend", methods=["POST", "GET"])
def recommend():

    # get ticker from the input
    symbol = request.form['symbol_input']
    symbol_info = yf.Ticker(symbol).info
    symbol_sector = symbol_info['sector']
    start_date = (datetime.date.today()+BDay(-250)).strftime("%Y-%m-%d")

    # get history data
    end_date = (datetime.date.today()).strftime("%Y-%m-%d")
    yesterday = (datetime.date.today()+BDay(-2)).strftime("%Y-%m-%d")
    year_start = pd.date_range(date(date.today().year, 1, 1).strftime('%Y-%m-%d'), date(date.today().year, 12, 1).strftime('%Y-%m-%d'), freq='BMS')[0].strftime('%Y-%m-%d')
    previous_year_end = pd.date_range(date(date.today().year-1, 12, 24).strftime('%Y-%m-%d'), date(date.today().year, 12, 31).strftime('%Y-%m-%d'), freq='BM')[0].strftime('%Y-%m-%d')
    min_date = min(start_date, year_start)
    first_BD = pd.date_range(date(date.today().year, 1, 1).strftime('%Y-%m-%d'), date(date.today().year, 12, 1).strftime('%Y-%m-%d'), freq='BMS')[0].strftime('%Y-%m-%d')
    symbol_history = pd.DataFrame(yf.Ticker(symbol).history(start=min_date))

    # get checkbox info
    check_box = request.form.getlist('mycheckbox')

    # update text
    flash("Displaying data for " + str(request.form['symbol_input']))

    # graph1
    fig1 = go.Figure(data=[go.Scatter(
        x = symbol_history['Close'].reset_index().Date,
        y = symbol_history['Close'].reset_index().Close,
        mode = 'lines',)
    ])

    fig1.update_layout(title_text='Trend of Close Prices for Last 250 Trading Days for ' + str(symbol))

    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    #graph2
    fig2 = go.Figure(data=[go.Histogram(
        x=symbol_history['Close'].reset_index().Close,
        nbinsx=int(round(len(symbol_history['Close'].reset_index().Close)/6)))
    ])

    fig2.update_layout(title_text='Histogram of Close Prices for Last 250 Trading Days for ' + str(symbol),
        bargap=0.15)

    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # high level table for stock perfroamnce
    symbol_dict = {}
    symbol_dict['Current Price'] = symbol_info['currentPrice']
    symbol_dict['Average'] =  "{:.2f}".format(np.mean(symbol_history.Close))
    symbol_dict['Std Dev'] =  "{:.2f}".format(np.std(symbol_history.Close))
    symbol_dict['Return Over 10 Days'] = "{:.2f}%".format(((symbol_info['currentPrice'] - symbol_history.Close[-10]) / symbol_history.Close[-10])*100)
    symbol_dict['Return Over 20 Days'] = "{:.2f}%".format(((symbol_info['currentPrice'] - symbol_history.Close[-20]) / symbol_history.Close[-20])*100)
    symbol_dict['YTD Return'] = "{:.2f}%".format(((symbol_info['currentPrice'] - symbol_history.Close.loc[previous_year_end]) / symbol_history.Close.loc[previous_year_end])*100)
    # check if median price available
    if symbol_info['targetMedianPrice'] == None:
        symbol_dict['Analyst Median Target Price'] = 0
        symbol_dict['Upside'] =  0
    else:
        symbol_dict['Analyst Median Target Price'] = symbol_info['targetMedianPrice']
        symbol_dict['Upside'] =  "{:.2f}%".format(((symbol_info['targetMedianPrice'] - symbol_info['currentPrice']) / symbol_info['currentPrice'])*100)
    # check if recommendation key available
    if symbol_info['recommendationKey']==None:
        symbol_dict['Analyst Recommendation'] = 'Nan'
    else:
        symbol_dict['Analyst Recommendation'] = symbol_info['recommendationKey']


    symbol_df = pd.DataFrame.from_dict(symbol_dict, orient='index').transpose()

    fig = go.Figure(data=[go.Table(
        columnwidth=[3, 3, 3, 3, 3, 3, 3, 3, 6],
        header=dict(values=list(symbol_df.columns),
                    fill_color='paleturquoise',
                    align='center'),
        cells=dict(values=[symbol_dict['Current Price'], symbol_dict['Average'], symbol_dict['Std Dev'], symbol_dict['Return Over 10 Days'], symbol_dict['Return Over 20 Days'],
                    symbol_dict['YTD Return'], symbol_dict['Analyst Median Target Price'], symbol_dict['Upside'], symbol_dict['Analyst Recommendation']],
                    fill_color='lavender',
                    align='center'))])

    fig.update_layout(title_text='Statistics for ' + str(symbol))

    tableOverviewJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # start all other tables and chrts as blank
    table1JSON = ""
    table2JSON = ""
    table3JSON = ""
    tableRecommendJSON = ""
    tablePredFundJSON = ""
    graphForecastJSON = ""
    graphOOSJSON = ""
    tableEuclidianJSON = ""

    # check if display index performance checkbox was ticked
    if '1' in check_box:

        # get sp_500 data
        data_sp500, data_dow, data_ftse100, data_ftse250, data_nasdaq = load_index_data()
        # filter only for constituents in the same symbol_sector
        data_sp500 = data_sp500[['Symbol', 'Security', 'Sector', 'GICS Sub-Industry']]
        # renma some sectors to align with yf.info['sector'] values
        if symbol_sector == 'Technology':
            symbol_sector = 'Information Technology'
        #data_sp500[data_sp500['Sector']=='Information Technology']='Technology'

        # calcualte returns for sp_500 symbol_history
        sp500_prices = yf.download(yfsi.tickers_sp500(), start=yesterday)['Close'][:2]
        sp500_dates = (sp500_prices.index).strftime('%d-%m-%y').tolist()
        sp500_prices.index = (sp500_prices.index).strftime('%d-%m-%y')
        sp500_returns = pd.DataFrame((sp500_prices.iloc[-1] - sp500_prices.iloc[0]) / sp500_prices.iloc[0])
        sp500_returns.columns = ['Daily Return']
        #sp500_returns['Daily Return %'] = pd.Series(["{0:.2f}%".format(val * 100) for val in sp500_returns['Daily Return']]).to_numpy()
        sp500_prices = sp500_prices[::-1].transpose()
        sp500_prices[sp500_dates[-1]] = sp500_prices[sp500_dates[-1]].apply('{:.2f}'.format)
        sp500_prices[sp500_dates[0]] = sp500_prices[sp500_dates[0]].apply('{:.2f}'.format)
        sp500_returns['Daily Return %'] = pd.Series(["{0:.2f}%".format(val * 100) for val in sp500_returns['Daily Return']]).to_numpy()
        # merge returns with index information
        sp500_df = pd.merge(data_sp500, sp500_prices[::-1], left_on=['Symbol'], right_index=True)
        sp500_df = pd.merge(sp500_df, sp500_returns, left_on=['Symbol'], right_index=True)
        sp500_df.sort_values(['Daily Return'], ascending=False, inplace=True)
        sp500_df.drop(columns=['Daily Return'], inplace=True)
        # get top and bottom performers
        sp500_df_top = sp500_df.head(10)
        sp500_df_bottom = sp500_df.tail(10)
        sp500_df = sp500_df[sp500_df['Sector'] == symbol_sector]

        # table of stocks in the same sector
        fig1 = go.Figure(data=[go.Table(
            columnwidth=[3, 5, 3, 8, 3, 3, 5],
            header=dict(values=list(sp500_df.columns),
                        fill_color='paleturquoise',
                        align='center'),
            cells=dict(values=[sp500_df.Symbol, sp500_df.Security, sp500_df.Sector, sp500_df['GICS Sub-Industry'],
                sp500_df[sp500_dates[-1]], sp500_df[sp500_dates[0]], sp500_df['Daily Return %']],
                        fill_color='lavender',
                        align='center'))])

        fig1.update_layout(title_text='S&P 500 Stocks Performance in Sector ' + str(symbol_sector))

        # table of top performers
        fig2 = go.Figure(data=[go.Table(
            columnwidth=[3, 5, 3, 8, 3, 3, 5],
            header=dict(values=list(sp500_df_top.columns),
                        fill_color='paleturquoise',
                        align='center'),
            cells=dict(values=[sp500_df_top.Symbol, sp500_df_top.Security, sp500_df_top.Sector, sp500_df_top['GICS Sub-Industry'],
                sp500_df_top[sp500_dates[-1]], sp500_df_top[sp500_dates[0]], sp500_df_top['Daily Return %']],
                        fill_color='lavender',
                        align='center'))])

        fig2.update_layout(title_text='Top 10 Daily Performers for S&P 500')

        # table of bottom performers
        fig3 = go.Figure(data=[go.Table(
            columnwidth=[3, 5, 3, 8, 3, 3, 5],
            header=dict(values=list(sp500_df_bottom.columns),
                        fill_color='paleturquoise',
                        align='center'),
            cells=dict(values=[sp500_df_bottom.Symbol, sp500_df_bottom.Security, sp500_df_bottom.Sector, sp500_df_bottom['GICS Sub-Industry'],
                sp500_df_bottom[sp500_dates[-1]], sp500_df_bottom[sp500_dates[0]], sp500_df_bottom['Daily Return %']],
                        fill_color='lavender',
                        align='center'))])

        fig3.update_layout(title_text='Bottom 10 Daily Performers for S&P 500')

        table1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
        table2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        table3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    # check if Forecast next 30 days is selected
    if '2' in check_box:

        # build the model
        n_steps = 80
        lstm = LstmModel()
        lstm.get_input_data(symbol)

        # create figure for plots
        fig = go.Figure()

        # check if there's enough data tgo train the model
        enough_data = ''
        try:
            lstm.normalize_data()
            lstm.train_test_split(val_size = 0.4, test_size = 0.2, n_steps=n_steps)
            lstm.lstm_model(lr = 0.0005, optimizer = 'adam', dropout=0.15, l1_size=40, l2_size=40, l3_size=40)
            lstm.fit_model(epochs=40, batch_size=64, verbose=1)

            # graph for out-of-samplem forecast for 1 day ahead
            plot_df = pd.DataFrame()
            plot_df['True Value'] = lstm.scaler.inverse_transform(lstm.y_test).reshape(-1)
            plot_df['LSTM Value'] = lstm.test_predict.reshape(-1)
            plot_df.set_index([lstm.df_original.iloc[-lstm.y_test.shape[0]:][:].reset_index()['Date']], inplace=True)

            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['True Value'], name='True Value',
                             line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['LSTM Value'], name = 'Predicted',
                                     line=dict(color='orange', width=2, dash='dot')))

        except:
            enough_data += 'Not enough data to train the model ... '

        fig.update_layout(title_text= enough_data + 'LSTM Out-of-sample Prediction for ' + str(symbol))

        graphOOSJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # create figure for plots
        fig = go.Figure()

        try:
            # predict the next 30 days
            predictions = [symbol_info['currentPrice']]
            predictions_dates = [date.today().strftime("%Y-%m-%d")]
            X_for_pred = lstm.X_test[-1]
            for x in range(1, 30):
                new_pred = lstm.lstm.predict(X_for_pred.reshape(-1, n_steps, 1))
                new_price = lstm.scaler.inverse_transform(new_pred)[0][0]
                predictions.append(new_price)
                predictions_dates.append((date.today()+BDay(x)).strftime("%Y-%m-%d"))
                X_for_pred = np.vstack([X_for_pred, new_pred])
                X_for_pred = X_for_pred[1:]

            # graph for predicted 5 days ahead
            fig = go.Figure(data=[go.Scatter(
                x = predictions_dates,
                y = predictions,
                mode = 'lines',)
            ])

        except:
            enough_data += 'Not enough data to train the model ... '

        fig.update_layout(title_text=enough_data + 'LSTM Prediction for the Next 30 Days for ' + str(symbol))

        graphForecastJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # check if recommendations where ticked
    if '3' in check_box:

        # get sp_500 data
        data_sp500, data_dow, data_ftse100, data_ftse250, data_nasdaq = load_index_data()
        # filter only for constituents in the same symbol_sector
        data_sp500 = data_sp500[['Symbol', 'Sector']]

        # RECOMMENDATIONS BASED ON SIMILARITY / MATRIX MULTIPLICATION
        attributes = open("contentRecommRanges.txt", "r")
        attr_list = attributes.read().splitlines()

        # import pre-stored df
        my_df = pd.read_csv('my_df', sep='\t')
        my_df = my_df.drop(['Unnamed: 0'], axis=1)
        my_df_extra = pd.read_csv('my_df_extra', sep='\t')
        my_df_extra = my_df_extra.drop(['Unnamed: 0'], axis=1)

        df_recommend = getRecommendations(my_df, my_df_extra, symbol, attr_list)
        # change colum names to include spaces
        new_cols = []
        for col in df_recommend.columns:
            new_cols.append(" ".join(re.split('(?=[A-Z])', col)).lstrip())

        df_recommend.columns = new_cols

        df_recommend = df_recommend.head(21)
        df_recommend = df_recommend.round(2)

        # merge with sp500 data to get Sector info
        df_recommend = data_sp500.merge(df_recommend, left_on = 'Symbol', right_on='Symbol')
        df_recommend.sort_values(by=['Similarity'], ascending=False, inplace=True)

        # table of stocks in the same sector
        fig = go.Figure(data=[go.Table(
            columnwidth=[4, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            header=dict(values=list(df_recommend.columns),
                        fill_color='paleturquoise',
                        align='center'),
            cells=dict(values=[df_recommend['Symbol'], df_recommend['Sector'],df_recommend['Security'], df_recommend['Similarity'], df_recommend['ebitda Margins'],
                            df_recommend['profit Margins'], df_recommend['gross Margins'], df_recommend['revenue Growth'], df_recommend['operating Margins'],
                            df_recommend['earnings Growth'], df_recommend['return On Assets'], df_recommend['debt To Equity'], df_recommend['return On Equity'],
                            df_recommend['enterprise To Ebitda'], df_recommend['price To Book'], df_recommend['price To Sales Trailing12 Months'], df_recommend['forward P E'], df_recommend['trailing P E']],
                        fill_color='lavender',
                        align='center'))])

        fig.update_layout(title_text='Content Based Recommendations Based on Matrix Multiplication for ' + str(symbol))

        tableRecommendJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # funk SVD prediction in case of missing fundamentals
        # create a copy of my_df_extra_pred to be used for funk SVD
        my_df_extra_pred = my_df_extra.copy()
        my_df_extra_pred.drop(['Security'], axis=1, inplace=True)
        my_df_extra_pred.set_index('Symbol', inplace=True)

        # create dataframe for the symbol in case it's not part of sp500 index and append it to the existing df
        if symbol not in my_df_extra.Symbol.tolist():

            # create dict of relevant values for ticker
            pref_stock_dict = {}
            input_list = []

            # create input list which will be list of fundamentals and brackets
            for i, x in enumerate(attr_list):
                input_list.append(attr_list[i].split(',')[0])

            # pre-populate the dictionary with nan where missing
            for key in input_list:
                pref_stock_dict[key] = np.nan

            # populate wtih actual fundamental values where available
            for key, value in symbol_info.items():
                if key in input_list:
                    pref_stock_dict[key] = value

            symbol_fundametals_df = pd.DataFrame.from_dict(pref_stock_dict, orient='index').transpose()
            symbol_fundametals_df.columns = new_cols[3:]
            # append this df to existing one to be used in SVD
            my_df_extra_pred.append(symbol_fundametals_df)
            # add right set_index
            new_index_list = my_df_extra_pred.index.tolist()[:-1]
            new_index_list.append(symbol)
            my_df_extra_pred.index = new_index_list

            # normalized the set and keep track of mean and Std
            mean_funkSVD = np.array(np.mean(my_df_extra_pred, axis=0))
            std_funkSVD = np.array(np.std(my_df_extra_pred, axis=0))
            my_df_extra_normalized = (my_df_extra_pred - mean_funkSVD) / std_funkSVD

            # perform funk SVD decomposition
            symbols_mat, features_mat = FunkSVD(np.matrix(my_df_extra_normalized), learning_rate=0.005, iters=300, latent_features=my_df_extra_pred.shape[1])

            # predict and re-scale
            predicted_fundametals = symbol_fundametals_df.copy()
            SVD_predicted_df = pd.DataFrame(np.dot(symbols_mat[-1], features_mat) * std_funkSVD + mean_funkSVD).transpose()

        # if symbol is part of S&P500 index, then import pre-calculated funk SVD data and predict from there
        else:
            # import pre-learned funk-svd
            predicted_df = pd.read_csv('predicted_df', sep='\t')
            predicted_df = predicted_df.set_index(['Symbol'])
            # import mean and std to re-scale (funk SVD runs on normalized data)
            mean_std_funkSVD = pd.read_csv('mean_std_funkSVD', sep='\t')
            mean_std_funkSVD = mean_std_funkSVD.set_index(['Statistic'])
            mean_funkSVD = np.array(mean_std_funkSVD.iloc[1])
            std_funkSVD = np.array(mean_std_funkSVD.iloc[0])
            # scale the prediction
            predicted_fundametals = pd.DataFrame(my_df_extra_pred.loc[symbol]).transpose()
            predicted_fundametals.columns = new_cols[3:]
            # re-scale based on mean and std
            SVD_predicted_df = pd.DataFrame(predicted_df.loc[symbol] * std_funkSVD + mean_funkSVD).transpose()

        SVD_predicted_df.columns = new_cols[3:]
        predicted_fundametals = predicted_fundametals.append(SVD_predicted_df)
        predicted_fundametals.insert(0, 'Type', ['Actual', 'Funk SVD Predicted'])
        predicted_fundametals = predicted_fundametals.round(2)

        # table of stocks in the same sector
        fig = go.Figure(data=[go.Table(
            columnwidth=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            header=dict(values=list(predicted_fundametals.columns),
                        fill_color='paleturquoise',
                        align='center'),
            cells=dict(values=[predicted_fundametals['Type'], predicted_fundametals['ebitda Margins'], predicted_fundametals['profit Margins'], predicted_fundametals['gross Margins'], predicted_fundametals['revenue Growth'],
                               predicted_fundametals['operating Margins'], predicted_fundametals['earnings Growth'], predicted_fundametals['return On Assets'], predicted_fundametals['debt To Equity'], predicted_fundametals['return On Equity'], predicted_fundametals['enterprise To Ebitda'],
                               predicted_fundametals['price To Book'], predicted_fundametals['price To Sales Trailing12 Months'], predicted_fundametals['forward P E'], predicted_fundametals['trailing P E']],
                        fill_color='lavender',
                        align='center'))])

        fig.update_layout(title_text='Predicted Fundamentals Based on S&P 500 Index for ' + str(symbol))

        tablePredFundJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', graph1JSON=graph1JSON, graph2JSON=graph2JSON, tableOverviewJSON=tableOverviewJSON,
        tableRecommendJSON=tableRecommendJSON, tablePredFundJSON=tablePredFundJSON, graphForecastJSON=graphForecastJSON,
        graphOOSJSON=graphOOSJSON, table1JSON=table1JSON, table2JSON=table2JSON, table3JSON=table3JSON)
