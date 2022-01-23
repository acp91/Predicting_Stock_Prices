# Predicting_Stock_Prices
Analyze historical performance and predict stock prices

# Project Definition

## Project Overeview
For the final project in Udacity Data Science Course I decided to recommend and predict stock prices based on user selection. I am primarily relying on two existing python packages, yfinance and yahoo_fin, that leverage yahoo finance APIs for retrieving stock prices and information. These APIs allow to query the most recent stock information on the market. Data provided through these APIs is the only data that I will work with.

## Problem Statement
The idea behind the project is simple: give some tools that could help user decide whether to buy, hold or sell a stock and to recommend similar stocks to the ones that he's already owning / interested in. While there's a lot that can still be improved (I address some of it later on) it's already an useful tool to get a general idea of how selected stock market is performing compared to the rest of the market.

I plan to solve this with a few different steps / methods, raging from simple to more advanced:
* Plot historical data / trend, plot histogram of prices and display current price in relation to median price from the analysts. If analysts' target price is much higher, it would mean stock is worth buying (undervalued); if analysts' target price is much lower, it would mean stock is worth selling (overvalued). If analysts' target price is close to current price, the stock should be held
* Show stock's performance in comparison to 1) the top performers of S&P 500 index and 2) stocks from S&P 500 index that are in the same Sector. How well stock performs relative to the overall market or the rest of its competitors further strengthens one's decision to buy/sell/hold a stock
* Train a machine learning model and use it to predict the future prices
* Use content based recommendation (matrix multiplication) to show most similar companies based on the fundamentals
* Use Funk SVD method to predict missing fundamentals (if any). Looking at stock's fundamentals can be an important factor when deciding whether a stock is overpriced or underpriced

In the end the user should decide on her/his own whether a stock is a buy/hold/sell. The app provides various angels to help with the decision, that is 1) analysts' recommendation, 2) predicting future prices based on historical trend and 3) comparing stock's fundamentals with other similar companies to see whether it's correctly valued.

## Metrics
I've implemented two separate models and for each of them I judged based on a different metric:
* Long short term memory (LSTM for short) neural network for predicting future stock prices. I've used mean squared error for model selection. Mean squared error is a good estimator in this case as I am not so interested in the magnitude of the predictions, just that the direction / trend is right. Trend would then provide meaningful insight into potential future profit & loss should one decide to buy/hold/sell a stock. If mean squared error is smaller it means that the prediction on average, was in the same direction as the actual price move (i.e. increase / decrease in the stock price)
* Funk SVD decomposition for predicting missing fundamentals' values. I've used a simple sum of absolute errors to measure model performance in this case. That is because I first standardize fundamentals to be in the range of 0-1. Certain fundamentals are ratios (from 0-100%) whereas others are on absolute level (e.g. certain margins can be 30 or 50). If I did not standardize the data first, the latter type of fundamentals would drive the overall error minimization function. Using absolute errors as opposed to squared errors further removes any bias towards higher-values of fundamentals. For example if errors for two standardized fundamentals are a=0.1 and b=0.5. The difference is of factor 5. If I squared them however, a2=0.01 and b2=0.25, the difference is of factor 25. But for me both errors are equally important
* Content Based Recommendation: I defined most similar companies based on how many fundamentals fall in the same bracket. The more fundamentals that two stocks have in common, the more similar they are. This is a generic logic for content based recommendation based on matrix multiplication and works well in my case as well

# Analysis

## Data Exploration
As part of the **Initial View** page (further explained below) the app provides general statistics about the selected stock price data, such as average, standard deviation, most recent returns.

## Data Visualization
As part of the **Initial View** page (further explained below) the app provides the plot of most recent trend and histogram of most recent prices (both over the course of the last 250 trading days).

# Methodology

## Data Preprocessing
I am relying on official data from Yahoo and therefore didn't run into many data issues. The only type of data preprocessing that I have performed is related to different Sector notations. In one of the options the app shows other stocks in the same Sector. Sector that is used for the main market indices (in my case S&P 500) does not always align with the Sector of a selected stock. One example I ran into is "Information Technology" for index vs "Technology" for stock.

## Implementation

### LSTM
  * Out of various different models that can be used to predict time series data (e.g. simple linear regression, AR, ARIMA) I've decided to rather use a deep-learning LSTM model. The reason for it is that LSTM model is designed to learn what information / past prices are worth keeping for predicting future prices and what information is not relevant. As discussed [here](https://datascience.stackexchange.com/questions/12721/time-series-prediction-using-arima-vs-lstm#:~:text=LSTM%20works%20better%20if%20we,not%20require%20setting%20such%20parameters) LSTM is a better choice than e.g. alternative ARIMA time series models for when we are dealing with a large amount of data (which gives deep leaning model enough information to learn on). Most of the well known stock have 10+ years of history which should be enough to train the model
  * The price of the stock rises and falls in time and the range between max and min can be substantial. Given prices for the same stock can be very different from each other, higher stock prices can have a larger impact on the error minimization function
  * Time series almost always contain trend that tends to result in spurious regression. This problem is known as non-stationarity and normal time series predictions are not meaningful; on non-stationary time series. It means that two time series seem to be correlated simply because they both exhibit a trend
  * In some cases there is not enough data to actually train the model

### Funk SVD
  * Funk SVD allows for prediction of missing values which can be very useful in my case as some of the less traded stocks might not have all the fundamentals available. Predicting missing fundamentals gives extra information on how correctly the stock is priced
  * As mentioned in *Analysis* section one problem I ran into was the fact that some fundamentals are ratios (from 0-100%) whereas others are on absolute level (e.g. certain margins can be 30 or 50). The results which were massively skewed towards higher valued absolute fundamentals
  ### Recommendation

### Content Based Recommendation
I ran into no particular problems when creating content based recommendation for a stock based on fundamental values. Below is the step-by-step process how it works:
  * It is based on S&P 500 data only
  * First I defined the relevant fundamentals in **contentRecommRanges.txt** file. A user can change this file (e.g. add new fundamentals, change ranges etc.). It is used to split fundamentals into different brackets which are then transformed into a matrix of 1s and 0s depending on whether the value of a stock for a fundamental falls in a certain bracket or not. User can change these brackets and re-create a new matrix if desired. Example: **ebitdaMargins, 0, 1, 11** would mean create 11 brackets between 1 and 0 for **ebitdaMargins** fundamental
    * First input the name of the fundamental. To see all available fundamentals, run yfinance.Ticker('GOOG').info (where GOOG can be replaced with any ticker)
    * The next 3 values represent lower bound, upper bound and number of brackets. Any value below lower bound is assigned to the lowest bracket and any value above upper bound is assigned to the highest bracket
  * Take all stocks for S&P 500 index and retrieve all the relevant fundamentals' values for them (relevant in this case means it's part of **contentRecommRanges.txt** file)
  * Produce associated matrices *my_df* and *my_df_extra* with **createDf()** function from **contentBasedRecom.py** python file and save them in the folder with the same names through **saveDataFrame()** function
    * my_df matrix contains information of 1s and 0s for when fundamentals of two compains fall in the same bracket
    * my_df_extra holds information of actual fundamental values and is later used to display the data in the app
  * In case selected ticker is part of S&P 500 index, no changes are needed for my_df and my_df_extra matrices
  * In case selected ticker is not part of S&P 500 index, append this ticker to both matrices
  * Multiply the row for the selected ticker with every other ticker in the matrix and sum. Given we populate matrix as 1 for where fundamentals fall in the same bracket, the higher the sum means the more similar the two companies are  

## Refinement

### LSTM
  * If there is not enough data to train the model the app will display a message stating there is not enough data and no predictions will be made. I did not decided for bootstrapping or different type of data-filling technique as I do not believe this would provide any benefit for predicting future stock prices
  * Before training the model I normalize the data to be between 0-1. By first normalizing the data I avoid any biases towards higher stock prices
  * Normally one would solve the problem of non-stationarity by first de-trending the data (i.e. removing the trend) and train the model on the newly formed time series. However as discussed [here](https://datascience.stackexchange.com/questions/24800/time-series-prediction-using-lstms-importance-of-making-time-series-stationar) LSTM model is designed to learn long-term dependencies and is therefore not affected by non-stationarity in the same way as linear regression or ARIMA models would be. Therefore I did not do any other transformations on my data

### Funk SVD
  * To avoid the problem mentioned above I first standardized the data to have all fundamentals between 0-1 before training Funk SVD model. This removed the bias where the error terms was primarily optimized for higher valued absolute fundamentals

# Results

## Model Evaluation and Validation

### LSTM
I've relied on some existing work as referenced in the *Reference* section in the end. I ran an LSTM model predicting 1-day ahead price looking for parameters that would minimize the sum of squared errors. After trying out different options these are the parameters my model is using:
  * Latest 80 prices to learn
  * Uses Adam optimizer (SGD approach would always flatten predictions too much therefore Adam seemed a better choice)
  * Validation sample size of 8%
  * Test sample size of 12%
  * Dropout rate of 5%
  * Uses 3 hidden layers of sizes 40
  * Learn rate of 0.1%

I've trained the model for various different parameters to find the optimal one:
![LSTM_optimize](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/LSTM_optimize.png)

All these parameters and more can be adjusted in app.py or LSTM.py python files.

### Funk SVD
Funk SVD is based on the same my_df_extra matrix that holds actual fundamental values for all the stocks. As we know Funk SVD works on missing values so in case some of the fundamentals are missing for the company, we can see what the predicted value should be.

I ran a few different scenarios to find funk SVD model that has the lowest sum of absolute errors. Learning rate of 0.005 was the most stable while the best result / smallest error was at 300 iterations:

![funk_SVD_optimize](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/funk_SVD_optimize.png)

## Justification
I've chosen both LSTM and Funk SVD models based on parameters that would minimize the sum of squared and absolute errors respectively.

# Conclusion

## Reflection

This project was overall very fun and interesting to me, regardless of the problems and amount of time it took me to finish it. I've tried to use all of the different techniques that we've covered throughout the Data Science course and create an app that can be actually used in practice. The app is based on official & reliable yahoo finance data that is always up-to date. At the same time this was the first project I ever did that covered both front-end and back-end; even though the front-end is very basic it was interesting to create something from scratch that can be shared with other users.

While I've done a lot of time series analysis during my work and studies, I've never used any deep-learning models in practice. Picking up LSTM model was a bit challenging at first but in the end I'm very glad I did as it gave me many new  insights into deep-learning models that I previously wasn't exposed to.

When I decided I wanted to work with financial data I didn't really know how to best use recommendation methodologies that we covered in the last section of the course. I spent quite some time on it and in the end I thought of using stocks' fundamentals and I'm quite satisfied with how it turned out.

## Improvement

### LSTM
There are a few places where LSTM model can be improved:
* To train one model it takes a few minutes, depending on the GPU (as the model relies on TensorFlow deep learning library). Therefore I did not fine-tune all possible parameters and only ran around 30 different model specifications. The model was also trained on IBM stock (as it has enough history for model to reliably learn). Ideally one would:
  * Train one model per stock of interest and have separate sets of parameters for each stocks
  * Fine tune the rest of the parameters and try other values (e.g. perform 100+ tests)
* Ideally we would train 1 model per forecasted day (i.e. 1 model for 1-day ahead, 1 model for 2-days ahead, ... , 1 model for 30-days ahead). Currently the app predicts the next 30 days based on 1 model only therefore each next step of forecast is less reliable. However there is not enough data to properly train all these models and even if there was it would take at least 1 hour to train all the models which would ruin user experience in this case. Once can still choose to do so on their side if there is a particular stock that they are interested in and has long enough history to reliably train all separate models

### Funk SVD
* Funk SVD model could be further improved by fine-tuning parameters (learning rate and the number of steps). I ran around 20 simulations but ideally one would run 50+ optimizations to find best possible parameters

### Content Based Recommendation
* Content Based Recommendation (as well as Funk SVD) would be even better if we extend the portfolio of underlying stocks from S&P 500 only to also include DOJAI, FTSE and NASDAQ. All these datasets can be retrieved with **load_index_data** function from **finRecomm.py** file. I have not done that as it would delay the amount of time it takes for user to see recommendations and at the same time S&P 500 is still best proxy for overall market performance
* To get even more accurate predictions, one could further refine relevant fundamentals in **contentRecommRanges.txt** file and create even more brackets for even more fundamentals

# Folder Structure
Folder structure is as follows:
* app Folder: parent Folder
  * lib Folder: library containing additional python scripts that are used
    * contentBasedRecom.py: used to create matrices for matrix based recommendation based on S&P 500 Index
    * finRecomm.py: imports some of the index data
    * LSTM.py: imports relevant data for the selected stock, builds LSTM model and can be used to predict future prices
  * static Folder: contains subfolders for the Flask app
    * css: css formatting for the webpage
    * images: images for the webpage
  * template Folder: contains index.html file which is used to render the webpage for Flask app
* images_git Folder: folder with images for GitHub repository

Additionally app Folder contains the following files:
* app.py: main python script for the Flask app
* contentRecommRanges.txt: txt file that serves as input to contentBasedRecom.py
* my_df & my_df_extra: outputs of function **createDf()** from **contentBasedRecom.py**. As fundamentals are only updated quarterly, I ran the process once and save the underlying data to be used for any stock
* requirements.txt: package requirements to run the Flask app
* Jupyter Notebooks (there are 3 Jupyter notebooks in part of the repo):
  * *contentBasedRecomm.ipynb*: shows step-by-step process how I trained Funk SVD recommendation and can also be used to try out whether different parameters work better
  * *LSTM_optimize.ipynb*: imports and trains LSTM model for different parameters
  * *createFundamentalsDfs.ipynb*: shows how to re-created data frames with fundamentals if needed. As mentioned above users can change what fundamentals they find relevant and what brackets they should have in **contentRecommRanges.txt** file

# App Summary
The Flask app that I have created allows the user to input a ticker of choice (official financial ticker, e.g. GOOG for google, AAPL for Apple and so on). Based on the user selection the app will:
* Plot historical prices
* Show analysts' recommendations and return over recent period
* Additionally allow the user to:
  * Display S&P 500 performance for the same sector and daily top 10/bottom 10 performing Stocks
  * Predict the next 30 days (if enough data available) based on LSTM model
  * Show most similar stocks / recommendations from the S&P 500 index based on similarity matrix where the input are stock fundamentals (price to book, ebidta margin etc.). It will also perform Funk SVD decomposition to cover cases where some of the fundamentals are missing)

# Running the App
To run the app, open cmd / conda Navigator, navigate into the app Folder and run command "flask app". You should get the following message:

![cmd_rn](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/cmd.png)

Copy the http path + '/inputSymbol' to get to the app's homepage. In my case it would look like **http://127.0.0.1:5000/inputSymbol**

## Initial View:
The homepage screen will look per below. Write the ticker of interest in the input field to get the basic view of the stock characteristics.

![InputScreen](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/InputScreen.png)

If you press **RECOMMEND** without ticking any of the tickboxes, this is the view you get:
* Trend of the closing prices for the selected ticker over the most recent 250 trading days
* Histogram of the closing prices for the selected ticker over the most recent 250 trading days
* Main statistics, such as current price, return over a few periods of interest, target analyst price, upside (return based on target analyst price and current price) and the current analyst recommendation

![DefaultView](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/DefaultView.png)

All the data that is shown here is pulled using yfinance and yahoo_fin packages for Python. The idea behind the **Initial View** screen is to give a rough idea to the user how the stock price has been performing and what would be the right move (i.e. whether to buy, hold or sell the stock).

Each of the tickboxes provides further information to the user on whether or not the stock is currently a buy, hold or sell as well as what might be other stocks of interest given the selected stock.

## Display S&P 500 Index Performance
*This selection will the delay response by ~15-30 seconds as it needs to download data for the entire S&P 500 index.*

User can additional tick any number of tickboxes. If **Display S&P 500 Index Performance** is ticked, the following information will be available:
* Performance of the stocks within the same Sector for S&P 500 Stocks
* Top 10 daily performing stocks for S&P 500 (i.e. highest daily returns)
* Bottom 10 daily performing stocks for S&P 500 (i.e. worst daily loses)

![ShowIndexInfo](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/ShowIndexInfo.png)

These additional information can provide more clarity and give a more realistic view of how selected stock is performing. Is it performing well for its sector? How is it performing compared to the overall market? Of course this type of information is more relevant for larger companies as we are comparing it to the S&P 500 stocks. While this is something that could be improved for the app (i.e. show performance of other indices as well), it can still be useful for comparison purposes even when low or mid cap stock is selected.

## Forecast Prices
*This selection will delay response by up to 1-1.5 minute as it needs to train an LSTM model.*

If **Forecast the Next 30 Days** is ticked, the following information will be available:
* Graph showing actual data and out-of-sample predictions for 1-day ahead prices
* Graph showing predictions for the next 30 days based on the model
* If not enough data is available, graph will display less data and inform the user for which days forecast was not possible

![Predictions](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/Predictions.png)

## Show Recommendations
*This selection will delay response by ~30-45 seconds as it perform various matrix multiplications and retrieve fundamentals for the selected stock.*

If **Show Recommendations** is ticked, the following information will be available:
* Table showing the top 20 most similar companies based on fundamentals
* Table showing actual fundamental values and predicted fundamental values based on Funk SVD decomposition

![Recommendations](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/Recommendations.png)

# References
## Building Flask App
* https://www.youtube.com/watch?v=6plVs_ytIH8&t=90s&ab_channel=PythonSimplified
* https://towardsdatascience.com/web-visualization-with-plotly-and-flask-3660abf9c946
* https://www.youtube.com/watch?v=B97qWOUvlnU&ab_channel=CodeWithPrince
* https://plotly.com/python/table-subplots/
* https://stackoverflow.com/questions/50188840/plotly-create-table-with-rowname-from-pandas-dataframe
* https://www.w3schools.com/howto/howto_css_custom_checkbox.asp
* https://www.youtube.com/watch?v=_sgVt16Q4O4&t=131s&ab_channel=PrettyPrinted

## Building LSTM Model
* https://en.wikipedia.org/wiki/Long_short-term_memory
* https://www.analyticsvidhya.com/blog/2021/05/stock-price-prediction-and-forecasting-using-stacked-lstm/
* https://datascience.stackexchange.com/questions/24800/time-series-prediction-using-lstms-importance-of-making-time-series-stationary
* https://www.simplilearn.com/tutorials/machine-learning-tutorial/stock-price-prediction-using-machine-learning
* https://www.youtube.com/watch?v=H6du_pfuznE&ab_channel=KrishNaik
* https://medium.com/@canerkilinc/selecting-optimal-lstm-batch-size-63066d88b96b#:~:text=By%20experience%2C%20in%20most%20cases,based%20on%20the%20performance%20observation.
