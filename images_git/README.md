# Predicting_Stock_Prices
Analyze historical performance and predict stock prices

For the final project in Udacity Data Science Course I decided to recommend and predict stock prices based on user selection. The app that I have created allows the user to input a ticker of choice (official financial ticker, e.g. GOOG for google, AAPL for Apple and so on). Based on the user selection the app will:
* plot historical prices
* show analysts' recommendations and return over recent period
* additionally allow the user to:
  * display S&P 500 performance for the same sector and daily top 10/bottom 10 performing Stocks
  * predict the next 5 days (if enough data available) based on LSTM model
  * show most similar stocks / recommendations from the S&P 500 index based on similarity matrix where the input are stock fundamentals (price to book, ebidta margin etc.). It will also perform Funk SVD decomposition in case some of the fundamentals are missing)

Folder is structured as follows:
1) Data: contains underlying disaster_categories.csv and disaster_messages.csv file (this is raw underlying data). process_data.py contains script that loads both data sets, merges them, cleans the merged data set and saves it in a new sqlite DB called DisasterResponse. Folder also contains ETL_Pipeline_Preparation.ipynb jupyter notebook with step by step process on data clean-up process
2) Model: contains train_classifier.py script that loads data from DisasterResponse DB and runs AdaBoostClassifier to classify tweets in the data. Code allows to optimize parameter selection through GridSearchCV functionality. For ease of use, pre-trained model is available within the Models folder called AdaBoostClassifier_model.pkl. Folder also contains "ML)Pipeline_Preparation.ipynb jupyter notebook with step by stpe process on data modelling. For reference, this is the pipeline/gridsearch used for the classifier:
