# Predicting_Stock_Prices
Analyze historical performance and predict stock prices

## Introduction to the Project
For the final project in Udacity Data Science Course I decided to recommend and predict stock prices based on user selection. The Flask app that I have created allows the user to input a ticker of choice (official financial ticker, e.g. GOOG for google, AAPL for Apple and so on). Based on the user selection the app will:
* plot historical prices
* show analysts' recommendations and return over recent period
* additionally allow the user to:
  * display S&P 500 performance for the same sector and daily top 10/bottom 10 performing Stocks
  * predict the next 5 days (if enough data available) based on LSTM model
  * show most similar stocks / recommendations from the S&P 500 index based on similarity matrix where the input are stock fundamentals (price to book, ebidta margin etc.). It will also perform Funk SVD decomposition in case some of the fundamentals are missing)

## Folder Structure
Folder structure is as follows:
* app Folder: parent Folder
  * lib Folder: library containing additional python scripts that are used
    * contentBasedRecom.py: used to create matrices for matrix based recommendation based on S&P 500 Index
    * finRecomm.py: imports some of the index data
    * LSTM.py: imports relevant data for the selected stock, builds LSTM model and can be used to predict future prices
  * static Folder: contains subfolders for the Flask app
    * css: css formating for the webpage
    * images: images for the webpage
  * template Folder: contains index.html file which is used to render the webpage for Flask app

Additionally app Folder contains the following files:
* app.py: main python script for the Flask app
* contentRecommRanges.txt: txt file that serves as input to contentBasedRecom.py. It is used to split fundamentals into different ranges which are then transformed into matrices of 1s and 0s depending on whether the value of a stock for a certain fundamental falls in a range or not. User can change these ranges and re-create a new matrix if desired. Example: **ebitdaMargins, 0, 1, 11**
  * first input the name of the fundametanl. To see all availabel fundamentals, run yahoo_fin.Ticker('GOOG').info (where GOOG can be replaced with any ticker)
  * the next 3 values represent lower bound, upper bound and number of brackets. Any value below lower bound is assigned to the lowest bracket and any value above upper bound is assigned to the highest bracket
* my_df & my_df_extra: outputs of function createDf() from contentBasedRecom.py. As fundamentals are only updated quarterly, I ran the process once and save the underlying data to be used for any stock
* requirements.txt: package requirements to run the Flask app

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

All the data that is shown here is pulled using yahoo_fin package for Python. The idea behind the **Initial View** screen is to give a rough idea to the user how the stock price has been performing and what would be the right move (i.e. whether to buy, hold or sell the stock).

Each of the tickboxes provides further information to the user on whether or not the stock is currently a buy, hold or sell as well as what might be other stocks of interest given the selected stock.

### Display S&P 500 Index Performance
User can additional tick any number of tickboxes. If **Display S&P 500 Index Performance** is ticked, the following information will be available:
* Performance of the stocks within the same Sector for S&P 500 Stocks
* Top 10 daily performing stocks for S&P 500 (i.e. highest daily returns)
* Bottom 10 daily performing stocks for S&P 500 (i.e. worst daily loses)

![ShowIndexInfo](https://github.com/acp91/Predicting_Stock_Prices/blob/main/images_git/ShowIndexInfo.png)

These additional information can provide more clarity and give a more realistic view of how selected stock is performing. Is it performing well for its sector? How is it performing compared to the overall market? Of course this type of information is more relevant for larger companies as we are comparing it to the S&P 500 stocks. While this is something that could be improved for the app (i.e. show performance of other indices as well), it can still be useful for comparison purposes even when low or mid cap stock is selected.
