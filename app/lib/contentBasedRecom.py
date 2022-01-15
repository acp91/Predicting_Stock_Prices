# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 12:48:22 2021

@author: Andre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import yahoo_fin
import datetime
import yahoo_fin.stock_info as yfsi

pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',1000)
#pd.options.display.float_format = '${:,.2f}'.format

# create dataframe with dummies for different levels of ratios
def createNames(list_input):
    names_list = []
    
    list_input = list_input.split(',')
    string = list_input[0]
    start = float(list_input[1])
    limit = float(list_input[2])
    steps = int(list_input[3])
    
    ranges = np.linspace(start, limit, steps)
    
    for x in range(ranges.shape[0]):
        names_list.append(string + '|' + str(round(ranges[x],2)))
        
    return names_list

# create a list of relevant columns
def colsForMatrix(list_input):
    cols_for_matrix = ['Symbol']
    
    for i, string in enumerate(list_input):
        cols_for_matrix.extend(createNames(string))
    
    return cols_for_matrix

# find the right column based on ratio value
def returnColumn(df, ratio, ratio_value):
    
    if ratio_value != None:
        # get list of matching columns
        matching = [s for s in df.columns if ratio in s]

        # find the right column for the value
        # check if greater than max:
        matched_col = None
    
        if ratio_value >= float(matching[-1][len(ratio)+1:]):
            matched_col = matching[-1]
        else:
            for i, x in enumerate(matching):
                check = float(matching[i][len(ratio)+1:])
                if (check > float(ratio_value) and float(ratio_value) > 0):
                    matched_col = matching[i-1]
                    break
                
        return matched_col
    
# create a df to be used for finding similar movies
def createDf(attr_list, data_index, sector=['Energy']):
    df = pd.DataFrame()
    df[colsForMatrix(attr_list)] = 0
    
    input_list = []
    
    for i, x in enumerate(attr_list):
        input_list.append(attr_list[i].split(',')[0])

    df_extra_info = pd.DataFrame()
    
    if sector != '':
        enumerate_list = enumerate(data_index[data_index['Sector'].isin(sector)].Symbol.tolist())
    else:
        enumerate_list = enumerate(data_index.Symbol.tolist())
    
    for i, symbol in enumerate_list:
        # get info for the ticker of interest
        pref_stock_info = yf.Ticker(symbol).info
        
        # create dict of relevant values for ticker
        pref_stock_dict = {}
    
        for key, value in pref_stock_info.items():
            if key in input_list:
                pref_stock_dict[key] = value
        
        for key, value in pref_stock_dict.items():
            matched_col = returnColumn(df, key, value)
            if matched_col != None:
                df.loc[i, matched_col] = 1
            df.loc[i, 'Symbol'] = symbol 
            
        df_extra_info = df_extra_info.append(pd.DataFrame.from_dict(pref_stock_dict, orient='index').transpose())
            
    df = df.fillna(0)
    df_extra_info.reset_index(inplace=True, drop=True)
    
    # relevant columns from index data
    data_index = data_index[['Symbol', 'Security']]    
        
    df_extra_info = pd.merge(pd.DataFrame(df.Symbol),df_extra_info , left_index=True, right_index=True)
    df_extra_info = pd.merge(df_extra_info, data_index, left_on='Symbol', right_on='Symbol')
    
    cols = df_extra_info.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_extra_info = df_extra_info[cols]
    
    return df, df_extra_info

def getRecommendations(df, df_extra, ticker, attr_list):
    
    # check if ticker exists in current df
    if not ticker in df.Symbol.tolist():
        
        df_ticker = pd.DataFrame()
        df_ticker[colsForMatrix(attr_list)] = 0        
        
        # get info for the ticker of interest
        pref_stock_info = yf.Ticker(ticker).info
        
        # create dict of relevant values for ticker
        pref_stock_dict = {}
        
        input_list = []
           
        for i, x in enumerate(attr_list):
            input_list.append(attr_list[i].split(',')[0])        
        
        for key in input_list:
            pref_stock_dict[key] = np.nan
            
        for key, value in pref_stock_info.items():
            if key in input_list:
                pref_stock_dict[key] = value
         
        for key, value in pref_stock_dict.items():
            matched_col = returnColumn(df, key, value)
            if matched_col != None:
                df_ticker.loc[i, matched_col] = 1
            df_ticker.loc[i, 'Symbol'] = ticker    
           
        df_ticker = df_ticker.fillna(0)
        df = df.append(df_ticker)
        df_extra = df_extra.append(pd.DataFrame.from_dict(pref_stock_dict, orient='index').transpose())
        df_extra = df_extra.reset_index(drop=True)
        df_extra.loc[df_extra.shape[0]-1, 'Symbol'] = ticker
        df_extra.loc[df_extra.shape[0]-1, 'Security'] = pref_stock_info['longName']       
        
        mat_prod = np.dot(np.array(df_ticker.iloc[:, 1:]), np.array(df.iloc[:, 1:].transpose()))
        mat_prod = mat_prod.reshape(-1)

    else:
    
        # get row based on ticker
        ticker_row = df[df['Symbol']==ticker].index[0]
        mat_prod = np.dot(np.array(df.iloc[ticker_row, 1:]), np.array(df.iloc[:, 1:].transpose()))
        
    # list of sorted recommendations indices
    sorted_indices = np.argsort(mat_prod)[::-1]
    
    # calculate similarities
    row_ticker = sorted_indices[0]
    value_ticker = mat_prod[row_ticker]
    ticker_prod = mat_prod
    similarities = []
    
    for x in sorted_indices:
        similarities.append(ticker_prod[x] / value_ticker)
    
    #print(similarities)
    
    new_df = df_extra.reindex(index = sorted_indices.tolist())
    new_df['Similarity'] = similarities
    
    cols = new_df.columns.tolist()
    cols = cols[:2] + cols[-1:] + cols[2:-1]
    new_df = new_df[cols]    
    new_df.sort_values(by=['Similarity'], ascending=False, inplace=True)
    
    # drop row with the actual symbol
    # new_df = new_df.drop(new_df[new_df['Symbol']==ticker].index[0])
    
    return new_df

def FunkSVD(fundamentals_mat, latent_features=14, learning_rate=0.0001, iters=100):
    '''
    This function performs matrix factorization using a basic form of FunkSVD with no regularization
    
    INPUT:
    ratings_mat - (numpy array) a matrix with users as rows, movies as columns, and ratings as values
    latent_features - (int) the number of latent features used
    learning_rate - (float) the learning rate 
    iters - (int) the number of iterations
    
    OUTPUT:
    user_mat - (numpy array) a user by latent feature matrix
    movie_mat - (numpy array) a latent feature by movie matrix
    '''
    
    # Set up useful values to be used through the rest of the function
    n_symbols = fundamentals_mat.shape[0] # number of rows in the matrix
    n_features = fundamentals_mat.shape[1] # number of movies in the matrix
    num_comb = n_symbols * n_features # total number of ratings in the matrix
    
    # initialize the user and movie matrices with random values
    # helpful link: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html
    symbols_mat = np.random.rand(n_symbols, latent_features) # user matrix filled with random values of shape user x latent 
    features_mat = np.random.rand(latent_features, n_features) # movie matrix filled with random values of shape latent x movies
    
    # initialize sse at 0 for first iteration
    sse_accum = 0
    
    # header for running results
    print("Optimization Statistics")
    print("Iterations | Mean Squared Error ")
    
    # for each iteration
    for iteration in range(iters):
        # update our sse
        old_sse = sse_accum
        sse_accum = 0
        
        # For each user-movie pair
        for symbol in range(n_symbols):
            for feature in range(n_features):
                # print('This is iteration {}, for user {} and movie {}.'.format(iteration, user, movie))
                
                # if the rating exists
                f_value = fundamentals_mat[symbol][0, feature]
                if not pd.isna(f_value):
                    # print('user {}, movie {}, rating {}'.format(user, movie, rating))
                    
                    # compute the error as the actual minus the dot product of the user and movie latent features
                    error = f_value - np.dot(symbols_mat[symbol], features_mat[:, feature])

                    # Keep track of the total sum of squared errors for the matrix
                    sse_accum =+ error
                    
                    # update the values in each matrix in the direction of the gradient
                    updated_symbols_mat = symbols_mat[symbol] + learning_rate * 2 * error * features_mat[:, feature]
                    symbols_mat[symbol] = updated_symbols_mat
                    updated_features_mat = features_mat[:, feature] + learning_rate * 2 * error * symbols_mat[symbol]
                    features_mat[:, feature] = updated_features_mat

        # print results for iteration
        try: 
            print('Step: {}, error is: {}.'.format(iteration, error))
        except:
            print('Rating was nan.')
        
    return symbols_mat, features_mat 
    
    
    












