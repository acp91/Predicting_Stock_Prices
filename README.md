# Predicting_Stock_Prices
Analyze historical performance and predict stock prices

# Introduction to the Project
For the final project in Udacity Data Science Course I decided to recommend and predict stock prices based on user selection. The Flask app that I have created allows the user to input a ticker of choice (official financial ticker, e.g. GOOG for google, AAPL for Apple and so on). Based on the user selection the app will:
* plot historical prices
* show analysts' recommendations and return over recent period
* additionally allow the user to:
  * display S&P 500 performance for the same sector and daily top 10/bottom 10 performing Stocks
  * predict the next 5 days (if enough data available) based on LSTM model
  * show most similar stocks / recommendations from the S&P 500 index based on similarity matrix where the input are stock fundamentals (price to book, ebidta margin etc.). It will also perform Funk SVD decomposition in case some of the fundamentals are missing)

# Folder Structure
Folder structure is as follows:
* app: parent Folder
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
* contentRecommRanges.txt: txt file that serves as input to contentBasedRecom.py. It is used to split fundamentals into different ranges which are then transformed into matrices of 1s and 0s depending on whether the value of a stock for a certain fundamental falls in a range or not. User can change these ranges and re-create a new matrix if desired
* my_df & my_df_extra: outputs of function createDf() from contentBasedRecom.py. As fundamentals are only updated quarterly, I ran the process once and save the underlying data to be used for any stock
* requirements.txt: package requirements to run the Flask app
