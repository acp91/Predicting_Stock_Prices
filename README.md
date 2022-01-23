# Predicting_Stock_Prices
Analyze historical performance and predict stock prices

## Introduction to the Project
For the final project in Udacity Data Science Course I decided to recommend and predict stock prices based on user selection. I am primarily relying on two existing python packages, yfinance and yahoo_fin, that leverage yahoo finance APIs for retrieving stock prices and information. These APIs allow to query the most recent stock information on the market.

The idea behing the project is simple: give some tools that could help user decide whether to buy, hold or sell a stock and to recommend similar stocks to the ones that he's already owning / lookin into. While there's a lot that can still be improved (I address this further towards the end), it's already an useful tool to get a general idea of how selected stock market is performing compared to the rest of the market.

The Flask app that I have created allows the user to input a ticker of choice (official financial ticker, e.g. GOOG for google, AAPL for Apple and so on). Based on the user selection the app will:
* Plot historical prices
* Show analysts' recommendations and return over recent period
* Additionally allow the user to:
  * Display S&P 500 performance for the same sector and daily top 10/bottom 10 performing Stocks
  * Predict the next 5 days (if enough data available) based on LSTM model
  * Show most similar stocks / recommendations from the S&P 500 index based on similarity matrix where the input are stock fundamentals (price to book, ebidta margin etc.). It will also perform Funk SVD decomposition in case some of the fundamentals are missing)

## Data Analysis
As part of the **Initial View** page (further explained below) the app provides some plots (time series plot and histogram) and general statistics about the selected stock price data, such as average, standard deviation, most recent returns and so on.

## Methodology
I am relying on official data from Yahoo and therefore didn't run into many data issues. The only type of data preprocessing that I have performed is related to different Sector notations. In one of the options the app shows other stocks in the same Sector. Sector that is used for the main market indices (in my case S&P 500) does not always align with the Sector of a selected stock. One example I ran into is "Information Technology" for index vs "Technology" for stock.

### Recommendation
For recommending similar stocks to the user I've performed content based recommendation by using matrix multiplication and funk SVD for predicting missing fundamentals. All these processes are based on fundamentals of stocks.

Top 20 most similar companies are selected in the following way:
* It is based on S&P 500 data only. It could be easily extended to include other major indices (e.g. DOWJ, FTSE) as yahoo_fin packages covers other indices as well, but for the purpose of this app I used S&P 500 only
* First I defined the relevant fundamentals in **contentRecommRanges.txt** file. A user can change this file (e.g. add new fundamentals, change ranges etc.). It is used to split fundamentals into different brackets which are then transformed into matrices of 1s and 0s depending on whether the value of a stock for a certain fundamental falls in a certain bracket or not. User can change these brackets and re-create a new matrix if desired. Example: **ebitdaMargins, 0, 1, 11** would mean create 11 brackets between 1 and 0 for **ebitdaMargins** fundamental
  * First input the name of the fundamental. To see all available fundamentals, run yfinance.Ticker('GOOG').info (where GOOG can be replaced with any ticker)
  * The next 3 values represent lower bound, upper bound and number of brackets. Any value below lower bound is assigned to the lowest bracket and any value above upper bound is assigned to the highest bracket
* Take all stocks for S&P 500 index and retrieve all the relevant fundamentals' values for them (relevant in this case means it's part of **contentRecommRanges.txt** file)
* Produce associated matrices *my_df* and *my_df_extra* with **createDf()** function from **contentBasedRecom.py** python file and save them in the folder with the same names through **saveDataFrame()** function
  * my_df matrix contains information of 1s and 0s for when fundamentals of two compains fall in the same bracket
  * my_df_extra holds information of actual fundamental values and is later used to display the data in the app
* In case selected ticker is part of S&P 500 index, no changes are needed for my_df and my_df_extra matrices
* In case selected ticker is not part of S&P 500 index, append this ticker to both matrices
* Multiply the row for the selected ticker with every other ticker in the matrix and sum. Given we populate matrix as 1 for where fundamentals fall in the same bracket, the higher the sum means the more similar the two companies are

Funk SVD is based on the same my_df_extra matrix that holds actual fundamental values for all the stocks. As we know Funk SVD works on missing values so in case some of the fundamentals are missing for the company, we can see what the predicted value should be.

I ran a few different scenarios to find funk SVD model that has the lowest error. Learning rate of 0.005 was the most stable while the best result / smallest error was at 300 iterations:

![funk_SVD_optimize](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/funk_SVD_optimize.png)

### Prediction
Out of various different models that can be used to predict time series data (e.g. simple linear regression, AR, ARIMA) I've decided to rather use a deep-learning based model called *Long-short-term-memory* or *LSTM* for short. The reason for it is that LSTM model is designed to learn what information / past prices are worth keeping for predicting future prices and what information is not relevant. As discussed [here](https://datascience.stackexchange.com/questions/12721/time-series-prediction-using-arima-vs-lstm#:~:text=LSTM%20works%20better%20if%20we,not%20require%20setting%20such%20parameters) LSTM is a better choice than e.g. alternative ARIMA time series models for when we are dealing with a large amount of data (which gives deep leaning model enough information to learn on).

I've relied on some existing work as referenced in the *Reference* section in the end.

I ran an LSTM model predicting 1-day ahead price. After trying out different options these are the parameters my model is using:
* Latest 15 prices to learn. I did not use a longer sequence for learning as it would limit the number of stocks for which I could make all 5 predictions
* Uses Adam optimizer. SGD did not converge in most of the examples that I run
* Validation sample size of 8%
* Test sample size of 12%
* Dropout rate of 5%
* Uses 3 hidden layers of sizes 30
* Learn rate of 0.1%

I've used mean squared error for model selection as well as keeping in mind that complicated models that would take a long time to run would ruin some of the user experience. With the model I have specified above it takes around 1 minute to train and display the data (but it will depend on users GPU as the code is relying on Tensor Flow library). Mean squared error is a good estimator in this case as I am not so interested in the magnitude of the predictions, just that the direction / trend is right. If mean squared error is smaller, it means that the prediction, on average, was in the same direction as the actual price move (i.e. increase / decrease in the stock price). I ran the tests for an IBM stock as it has a long history of prices.

![LSTM_optimize](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/LSTM_optimize.png)

All these parameters and more can be adjusted in app.py or LSTM.py python files.

## Folder Structure
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

Additionally app Folder contains the following files:
* app.py: main python script for the Flask app
* contentRecommRanges.txt: txt file that serves as input to contentBasedRecom.py
* my_df & my_df_extra: outputs of function **createDf()** from **contentBasedRecom.py**. As fundamentals are only updated quarterly, I ran the process once and save the underlying data to be used for any stock
* requirements.txt: package requirements to run the Flask app
* Jupyter Notebooks (there are 3 Jupyter notebooks in part of the repo):
  * *contentBasedRecomm.ipynb*: shows step-by-step process how I trained Funk SVD recommendation and can also be used to try out whether different parameters work better
  * *LSTM_optimize.ipynb*: imports and trains LSTM model for different parameters
  * *createFundamentalsDfs.ipynb*: shows how to re-created data frames with fundamentals if needed. As mentioned above users can change what fundamentals they find relevant and what brackets they should have in **contentRecommRanges.txt** file

## Running the App
To run the app, open cmd / conda Navigator, navigate into the app Folder and run command "flask app". You should get the following message:

![cmd_rn](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/cmd.png)

Copy the http path + '/inputSymbol' to get to the app's homepage. In my case it would look like **http://127.0.0.1:5000/inputSymbol**

### Initial View:
The homepage screen will look per below. Write the ticker of interest in the input field to get the basic view of the stock characteristics.

![InputScreen](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/InputScreen.png)

If you press **RECOMMEND** without ticking any of the tickboxes, this is the view you get:
* Trend of the closing prices for the selected ticker over the most recent 250 trading days
* Histogram of the closing prices for the selected ticker over the most recent 250 trading days
* Main statistics, such as current price, return over a few periods of interest, target analyst price, upside (return based on target analyst price and current price) and the current analyst recommendation

![DefaultView](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/DefaultView.png)

All the data that is shown here is pulled using yfinance and yahoo_fin packages for Python. The idea behind the **Initial View** screen is to give a rough idea to the user how the stock price has been performing and what would be the right move (i.e. whether to buy, hold or sell the stock).

Each of the tickboxes provides further information to the user on whether or not the stock is currently a buy, hold or sell as well as what might be other stocks of interest given the selected stock.

### Display S&P 500 Index Performance
*This selection will the delay response by ~15-30 seconds as it needs to download data for the entire S&P 500 index.*

User can additional tick any number of tickboxes. If **Display S&P 500 Index Performance** is ticked, the following information will be available:
* Performance of the stocks within the same Sector for S&P 500 Stocks
* Top 10 daily performing stocks for S&P 500 (i.e. highest daily returns)
* Bottom 10 daily performing stocks for S&P 500 (i.e. worst daily loses)

![ShowIndexInfo](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/ShowIndexInfo.png)

These additional information can provide more clarity and give a more realistic view of how selected stock is performing. Is it performing well for its sector? How is it performing compared to the overall market? Of course this type of information is more relevant for larger companies as we are comparing it to the S&P 500 stocks. While this is something that could be improved for the app (i.e. show performance of other indices as well), it can still be useful for comparison purposes even when low or mid cap stock is selected.

### Forecast Prices
*This selection will delay response by up to 1-1.5 minute as it needs to train an LSTM model.*

If **Forecast the Next 30 Days** is ticked, the following information will be available:
* Graph showing actual data and out-of-sample predictions for 1-day ahead prices
* Graph showing predictions for the next 30 days based on the model. Ideally we would train 1 model per forecasted day (i.e. 1 model for 1-day ahead, 1 model for 2-days ahead, ... , 1 model for 30-days ahead); however there is not enough data to properly train all these models and even if there was it would take at least 30min to train all the models which would ruin user experience in this case. Once can still choose to do so on their  side if there is a particular stock that they are interested in
* If not enough data is available, graph will display less data and inform the user for which days forecast was not possible

![Predictions](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/Predictions.png)

### Show Recommendations
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
