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
    '''
    Creates a list of all possible fundamentals and their associated brackets

    Parameters
    ----------
    list_input : text file of the format "contentRecommRanges.txt" in the main folder
        List that defines relevant fundamentals and what brackets they should be split into

    Returns
    -------
    names_list : list
        List of all fundamentals and assocaited brackets. E.g. if input is PE_Ration, 0, 1, 2 it returns:
            PE_Ratio_0, PE_Ratio_0.5, PE_Ratio 1

    '''
    # create empty list to store all fundamentals and possible brackets
    names_list = []
    
    # read the text file
    list_input = list_input.split(',')
    # get the name of the fundamental
    string = list_input[0]
    # get the lower bound
    start = float(list_input[1])
    # get the upper bound
    limit = float(list_input[2])
    # get the number of steps
    steps = int(list_input[3])
    
    # create brackets based on start, limit and steps
    ranges = np.linspace(start, limit, steps)
    
    # concatenate  fundamental and all possible brackets
    for x in range(ranges.shape[0]):
        names_list.append(string + '|' + str(round(ranges[x],2)))
        
    return names_list

# create a list of relevant columns
def colsForMatrix(list_input):
    '''
    Create final list of columns for the matrix

    Parameters
    ----------
    list_input : list
        List created by createNames function.

    Returns
    -------
    cols_for_matrix : list
        All columns needed for the matrix.

    '''
    cols_for_matrix = ['Symbol']
    
    # extend the list
    for i, string in enumerate(list_input):
        cols_for_matrix.extend(createNames(string))
    
    return cols_for_matrix

# find the right column based on ratio value
def returnColumn(df, ratio, ratio_value):
    '''
    Function that finds in what column a given fundamental value falls into

    Parameters
    ----------
    df : dataframe
        DESCRIPTION.
    ratio : TYPE
        DESCRIPTION.
    ratio_value : TYPE
        DESCRIPTION.

    Returns
    -------
    matched_col : TYPE
        DESCRIPTION.

    '''
    
    if ratio_value != None:
        # get list of matching columns
        matching = [s for s in df.columns if ratio in s]

        # find the right column for the value
        # check if greater than max:
        matched_col = None
    
        # if value above max -> set to max bracket
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
    '''
    Creates dataframes for all stocks in an index (limited to sector if specified)

    Parameters
    ----------
    attr_list : text file
        Data in contentRecommRanges.txt file.
    data_index : dataframe
        yahoo finance index e.g. use yahoo_fin.stock_info.tickers_sp500(True)
    sector : list of strings, optional
        List of sectors of interest (index will be filtered just for the subset). The default is ['Energy'].

    Returns
    -------
    df : dataframe
        Dataframe of 1s and 0s depending on whether stocks fundamental falls in a certain range.
    df_extra_info : dataframe
        Dataframe of all stocks in an index and their fundamentals.

    '''
    
    # create a new dataframe that will hold 1s and 0s
    df = pd.DataFrame()
    # create a new dataframe that will hold fundamentals for each stock
    df_extra_info = pd.DataFrame()
    
    # assing columns based on colsForMatrix function
    df[colsForMatrix(attr_list)] = 0
    
    input_list = []
    
    # create list of fundamentals that are of interest
    for i, x in enumerate(attr_list):
        input_list.append(attr_list[i].split(',')[0])
    
    # create a list of all stocks from the index
    if sector != '':
        enumerate_list = enumerate(data_index[data_index['Sector'].isin(sector)].Symbol.tolist())
    else:
        enumerate_list = enumerate(data_index.Symbol.tolist())
    
    # loop through all the stocks
    for i, symbol in enumerate_list:
        # get info for the ticker of interest
        pref_stock_info = yf.Ticker(symbol).info
        
        # create dict of relevant values for ticker
        pref_stock_dict = {}
    
        for key, value in pref_stock_info.items():
            if key in input_list:
                pref_stock_dict[key] = value
        
        # find the column that fits stocks' fundamental value and populate it as 1 in df
        for key, value in pref_stock_dict.items():
            matched_col = returnColumn(df, key, value)
            if matched_col != None:
                df.loc[i, matched_col] = 1
            df.loc[i, 'Symbol'] = symbol 
        
        # add stocks' fundamentals to df_extra_info
        df_extra_info = df_extra_info.append(pd.DataFrame.from_dict(pref_stock_dict, orient='index').transpose())
            
    # fill nan values as 0 so we can use df matrix for multiplication (for content based recommendation)        
    df = df.fillna(0)
    df_extra_info.reset_index(inplace=True, drop=True)
    
    # relevant columns from index data
    data_index = data_index[['Symbol', 'Security']]    
        
    df_extra_info = pd.merge(pd.DataFrame(df.Symbol),df_extra_info , left_index=True, right_index=True)
    df_extra_info = pd.merge(df_extra_info, data_index, left_on='Symbol', right_on='Symbol')
    
    # flip columns so Symbol is first
    cols = df_extra_info.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_extra_info = df_extra_info[cols]
    
    return df, df_extra_info

def getRecommendations(df, df_extra, ticker, attr_list):
    '''
    Get ordered recommendations for stocks (based on fundamentals)

    Parameters
    ----------
    df : dataframe
        df from createDf.
    df_extra : dataframe
        df_extra_info from createDf.
    ticker : string
        Official financial ticker, check yahoo finance if not sure.
    attr_list : text file
        Data in contentRecommRanges.txt file.

    Returns
    -------
    new_df : dataframe
        dataframe of recommendations ordered from most similar to least.

    '''
    
    # check if ticker exists in current df
    if not ticker in df.Symbol.tolist():
        
        df_ticker = pd.DataFrame()
        df_ticker[colsForMatrix(attr_list)] = 0        
        
        # get info for the ticker of interest
        pref_stock_info = yf.Ticker(ticker).info
        
        # create dict of relevant values for ticker
        pref_stock_dict = {}
        
        input_list = []
           
        # create list of fundamentals
        for i, x in enumerate(attr_list):
            input_list.append(attr_list[i].split(',')[0])        
        
        # create dictionary of stocks fundamentals
        for key in input_list:
            pref_stock_dict[key] = np.nan
            
        for key, value in pref_stock_info.items():
            if key in input_list:
                pref_stock_dict[key] = value
        
        # find out in which column stock's fundamental fits and populate it as 1
        for key, value in pref_stock_dict.items():
            matched_col = returnColumn(df, key, value)
            if matched_col != None:
                df_ticker.loc[i, matched_col] = 1
            df_ticker.loc[i, 'Symbol'] = ticker    
           
        # clean-up dataframes so they can be used in matrix multiplication
        df_ticker = df_ticker.fillna(0)
        df = df.append(df_ticker)
        df_extra = df_extra.append(pd.DataFrame.from_dict(pref_stock_dict, orient='index').transpose())
        df_extra = df_extra.reset_index(drop=True)
        df_extra.loc[df_extra.shape[0]-1, 'Symbol'] = ticker
        df_extra.loc[df_extra.shape[0]-1, 'Security'] = pref_stock_info['longName']       
        
        # matrix multiplication for content based recommendation
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
    
    # re-oreder colums
    cols = new_df.columns.tolist()
    cols = cols[:2] + cols[-1:] + cols[2:-1]
    new_df = new_df[cols]    
    # sort by similarity from most similar to least
    new_df.sort_values(by=['Similarity'], ascending=False, inplace=True)
    
    # drop row with the actual symbol
    # new_df = new_df.drop(new_df[new_df['Symbol']==ticker].index[0])
    
    return new_df

def FunkSVD(fundamentals_mat, latent_features=14, learning_rate=0.0001, iters=100):
    '''
    This function performs matrix factorization using a basic form of FunkSVD with no regularization

    Parameters
    ----------
    fundamentals_mat : dataframe
        A matrix with symbols as rows, fundamentals as columns and fundamentals as values.
    latent_features : int, optional
        Number of fundamentals. The default is 14.
    learning_rate : int, optional
        Learning rate for Funk SVD model. The default is 0.0001.
    iters : int, optional
        Number of steps to trian Funk SVD model. The default is 100.

    Returns
    -------
    symbols_mat : dataframe
        Left matrix from Funk SVD decomp.
    features_mat : dataframe
        Right matrix from Funk SVD decomp.

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
    
    
    












