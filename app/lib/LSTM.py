# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 19:43:13 2021

@author: Andre
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

class LstmModel():
    '''
    Class that allows for trianing an LSTM model - covers normalizaiton of data, splitting data into train, validaiton 
    and test sets, training the data and making predictions
    '''
    
    def get_input_data(self, ticker,
                       start='1990-01-01', 
                       end=(datetime.date.today()+datetime.timedelta(1)).strftime("%Y-%m-%d"),
                       interval='1d',
                       type='Close',
                       offset=1):
        '''
        

        Parameters
        ----------
        ticker : string
            Official financial ticker for the stock of interested. If unsure please check online e.g. yahoo finance.
        start : string, optional
            Start data for retrieving historical data for the stock. The default is '1990-01-01'.
        end : string, optional
            End data for retrieving historical data for the stock. The default is today + 1 day (to get full history).
        interval : string, optional
            Interval of prices to return. The default is '1d'. Check possible values for yfinance API
        type : string, optional
            Define type of prices to retrieve, i.e. Close, Open, High. The default is 'Close'.
        offset : int, optional
            Similar to interval parameter - however interval parameter doesnt cover all possible days.
            User could select random offset here, e.g. 8 days. The default is 1.

        Returns
        -------
        None. It stores original df as df_original and an array of prices in df

        '''
       
        # get ticker information
        self.ticker = ticker
        
        # retrive historical prices for the ticker
        df = yf.Ticker(self.ticker).history(start=start, end=end, interval=interval)
        # apply offset if specified
        df = df.iloc[::offset]    
        
        # store original data
        original_df = df
        self.df_original = df
        
        # store array of values
        df = df[type].values
        self.df = df
        
    def normalize_data(self):
        '''
        Normalize the data based on Sklearn minMaxScaler

        Returns
        -------
        None. Store transformed data info df_tr and scaler as scaler (this scaler is later used to inverse data)

        '''
        
        # create MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        
        # transform data
        df_tr = scaler.fit_transform(np.array(self.df).reshape(-1,1))
        
        # store transformed data and scaler
        self.df_tr = df_tr
        self.scaler = scaler
        
    def train_test_split(self, val_size = 0.4, test_size = 0.15, n_steps=15):
        '''
        Split the data into trian, validaiton and test sets

        Parameters
        ----------
        val_size : int, optional
            Size of val_size relative to test_size. The default is 0.4. E.g. if test size is 0.15, then validation size is
            0.4 * 0.15 = 0.06
        test_size : int, optional
            Size of test size (includes val_size as well). The default is 0.15. E.g. if validation size is 0.15, then 
            test size is (1- 0.4) * 0.15 = 0.09
        n_steps : int, optional
            Defines how many days will be used in each step of the prediction. The default is 15. This means previous 15 days
            are used for predicting the next price

        Returns
        -------
        None. Store X and y train, validaiton and test dataframes

        '''
        # dont shuffle the data during the split as it's TS data
        # split into train, validaiton and test set
        df_train, df_test = train_test_split(self.df_tr, test_size=test_size, random_state=None, shuffle=False)
        df_val, df_test = train_test_split(df_test, test_size=1-val_size, random_state=0, shuffle=False)
        
        # all create_df function to create dataframes for X and y variables for train, validation and test
        X_train, y_train = create_data(df_train, n_steps)
        X_val, y_val = create_data(df_val, n_steps)
        X_test, y_test = create_data(df_test, n_steps)
        
        # shape should be: number of entries, number of steps, 1 when entering LSTM model
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)        
        
        # store variables
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
    def lstm_model(self, lr = 0.005, optimizer = 'adam', dropout=0.1, l1_size=50, l2_size=40, l3_size=30):
        '''
        Build LSTM model

        Parameters
        ----------
        lr : int, optional
            Learning rate for the LSTM model. The default is 0.005.
        optimizer : string, optional
            Optimizer - can be either adam or SGD. The default is 'adam'.
        dropout : int, optional
            Dropout rate aftear each gradient iteration. The default is 0.1.
        l1_size : int, optional
            Size /number of ndoes in first hidden layer. The default is 50.
        l2_size : int, optional
            Size /number of ndoes in second hidden layer. The default is 40.
        l3_size : int, optional
            Size /number of ndoes in third hidden layer. The default is 30.

        Returns
        -------
        None. Store trained LSTM model

        '''
        
        #Building the LSTM Model
        lstm = Sequential()
        # first layer
        lstm.add(LSTM(l1_size, input_shape=(self.X_train.shape[1], 1), activation='tanh', return_sequences=True))
        lstm.add(Dropout(dropout))
        # second layer
        lstm.add(LSTM(l2_size, activation='tanh', return_sequences=True))
        lstm.add(Dropout(dropout))
        # third layer
        lstm.add(LSTM(l3_size, activation='tanh'))
        lstm.add(Dropout(dropout))
        # final layer / prediction
        lstm.add(Dense(1))
        
        # define th optimizer
        if optimizer == 'adam':
            optim = optimizers.Adam(learning_rate=lr)
        elif optimizer == 'SGD':
            optim = optimizers.SGD(learning_rate=lr, momentum=0.1, nesterov=False, name='SGD')
            
        # calculate loss for the optimizer    
        lstm.compile(loss='mean_squared_error', optimizer=optim) #, metrics=['accuracy'])        
        
        self.lstm = lstm
    
    def fit_model(self, epochs=40, batch_size=64, verbose=0):   
        '''
        Train LSTM model

        Parameters
        ----------
        epochs : int, optional
            Number of interations / passes through the deep-learning model. The default is 40.
        batch_size : TYPE, optional
            Batch size - number of predictions that go into the model at the same time to learn of any common patterns.
            The default is 64.
        verbose : int, optional
            Defines whether to print out loss at each epoch. The default is 0.

        Returns
        -------
        None. Store predicted data on train and test sets

        '''
        
        # fit the model
        self.lstm.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=epochs, batch_size=batch_size, verbose=verbose)
        # predict
        train_predict = self.lstm.predict(self.X_train)
        test_predict = self.lstm.predict(self.X_test)
        
        # transform back with scaler
        self.train_predict = self.scaler.inverse_transform(train_predict)
        self.test_predict = self.scaler.inverse_transform(test_predict)        
        
    def plot_pred_real(self, x_size=15, y_size=10, font_size=12):
        '''
        Plot actual and out-of-sample predictions

        Parameters
        ----------
        x_size : int, optional
            Length of x-axis for the chart. The default is 15.
        y_size : int, optional
            Length of the y-axis for the chart. The default is 10.
        font_size : int, optional
            Font size on the chart. The default is 12.

        Returns
        -------
        None. Plots the graph
        '''
        
        # create a df with test and pred values & dates
        plot_df = pd.DataFrame()
        plot_df['True Value'] = self.scaler.inverse_transform(self.y_test).reshape(-1)
        plot_df['LSTM Value'] = self.test_predict.reshape(-1)
        plot_df.set_index([self.df_original.iloc[-self.y_test.shape[0]:][:].reset_index()['Date']], inplace=True)
        
        #Predicted vs True Adj Close Value – LSTM
        plt.figure(figsize=(x_size, y_size))
        plt.rcParams['font.size'] = font_size
        plt.plot(plot_df['True Value'], label='True Value', color='blue', alpha=0.8, linestyle='--')
        plt.plot(plot_df['LSTM Value'], label='LSTM Value', color='orange', alpha=0.5, linewidth=3)
        #plt.xticks(dates)
        plt.title('Prediction by LSTM for {}'.format(self.ticker))
        plt.xlabel('Dates')
        plt.ylabel('USD')
        plt.legend()
        plt.show()
        
                
def create_data(data, step=1):
    '''
    

    Parameters
    ----------
    data : dataframe
        Data that will be split into X and y variables for training the model.
    step : int, optional
        Defines the step e.g. do you take every value, every second vlaue etc. The default is 1 (take every value).

    Returns
    -------
    TYPE np.array, np.array
        Returns two np.arrays, X and y

    '''
    X, Y = [], []
    for i in range(data.shape[0]-step):
        x_data = data[i:(i + step)]
        y_data = data[i + step]
        X.append(x_data)
        Y.append(y_data)
            
    return np.array(X), np.array(Y)        
        
        
        
        
        