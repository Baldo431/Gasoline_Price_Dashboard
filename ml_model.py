# Gas & Oil Linear Regression Model
import os
import pickle
import numpy as np
import pandas as pd
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

## Description of Preliminary Data Preprocessing
# Starting with crude oil data (1983 to present) and gas and diesel price data (1995 to 2021), the data are cleaned to fit into a linear regression machine learning model.
# Below the data are loaded into two dataframes (gas_df and crude_df). The cleaned gas price dataframe includes all formulations of retail gasoline and diesel prices in a MM/DD/YYYY format with samples from each month starting in January 1995 to January 2021. The cleaned crude oil dataframe is in the same format as the cleaned gas price dataframe: MM/DD/YYYY format with monthly samples from January 1995 to January 2021.
## Data Examining & Cleaning 

#A1: Weekly U.S. All Grades All Formulations Retail Gasoline Prices (Dollars per Gallon)
#A2: Weekly U.S. All Grades Conventional Retail Gasoline Prices (Dollars per Gallon)
#A3: Weekly U.S. All Grades Reformulated Retail Gasoline Prices (Dollars per Gallon)
#R1: Weekly U.S. Regular All Formulations Retail Gasoline Prices (Dollars per Gallon)
#R2: Weekly U.S. Regular Conventional Retail Gasoline Prices (Dollars per Gallon)
#R3: Weekly U.S. Regular Reformulated Retail Gasoline Prices (Dollars per Gallon)
#M1: Weekly U.S. Midgrade All Formulations Retail Gasoline Prices (Dollars per Gallon)
#M2: Weekly U.S. Midgrade Conventional Retail Gasoline Prices (Dollars per Gallon)
#M3: Weekly U.S. Midgrade Reformulated Retail Gasoline Prices (Dollars per Gallon)
#P1: Weekly U.S. Premium All Formulations Retail Gasoline Prices (Dollars per Gallon)
#P2: Weekly U.S. Premium Conventional Retail Gasoline Prices (Dollars per Gallon)
#P3: Weekly U.S. Premium Reformulated Retail Gasoline Prices (Dollars per Gallon)
#D1: Weekly U.S. No 2 Diesel Retail Prices (Dollars per Gallon)

hist_gas_df = pd.read_csv("Resources/data/ML_gas_price.csv")

# Dropping conventional and reformulated retail gas prices
hist_gas_df['date'] = hist_gas_df['Date']
hist_gas_df = hist_gas_df[['date', 'R1', 'M1', 'P1', 'D1']]

crude_df = pd.read_csv("Resources/data/ML_crude_price.csv")

# Dropping percent change and change of crude oil prices
crude_df = crude_df[['date', 'price']]

# Formatting crude_df to match hist_gas_df date format
def fix_date(bad_date):
    bad_date = str(bad_date)
    year, month, day = bad_date.split('-')
    day = day[0:2]
    return date(int(year), int(month), int(day))

crude_df['date'] = crude_df['date'].apply(fix_date) 

# Dropping 1983-1994 from crude_df
cuttoff_date = date(1994, 12, 31)
crude_df = crude_df.loc[crude_df['date'] > cuttoff_date]

# Lining up months and years in crude_df and gas_df
def date2str(dt):
    return dt.strftime('%m/%Y')

def drop_month(dt):
    month, day, year = dt.split('/')
    return f"{month}/{year}"

crude_df['date'] = crude_df['date'].apply(date2str)
hist_gas_df['date'] = hist_gas_df['date'].apply(drop_month)

# Omit 2022 data 
crude_df = crude_df.loc[0:454]

# Keep only first Month/Year instance for gas_df
seen_dates = set()
indices_to_remove = []
for idx, row in hist_gas_df.iterrows():
    _date = row['date']
    if _date in seen_dates:
        indices_to_remove.append(idx)
    else:
        seen_dates.add(_date)
hist_gas_df.drop(indices_to_remove, inplace=True)

# Linear Regression Machine Learning Model

## Description of Preliminary Feature Engineering and Preliminary Feature Selection
# To fit a linear regression model with crude oil and gas price data to establish and explore relationships between the data.

# Explanation of Model Choice, including Limitations and Benefits 
# This linear regression model is the best model because historically, crude oil prices and gas prices have a linear relationship, so there is no need to complicate the relationship with other, more complex models.

# X = crude oil price [=] 1/barrel
# Y = gas price [=] 1/gallon

def build_model(X, Y, filename):
    # Train Test Split
    # The data are split into 80% training and 20% testing.
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

    X_train = X_train.reshape((-1,1))
    X_test = X_test.reshape((-1,1))

    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train)

    # Save linear regression model
    pickle.dump(lin_reg_model, open('Resources/data/' + filename, 'wb'))


# Create model for each of the fuel types
X = np.array(crude_df['price'])

Y = hist_gas_df['R1'].to_numpy()
build_model(X, Y, 'reg_model.sav')

Y = hist_gas_df['M1'].to_numpy()
build_model(X, Y, 'mid_model.sav')

Y = hist_gas_df['P1'].to_numpy()
build_model(X, Y, 'prm_model.sav')

Y = hist_gas_df['D1'].to_numpy()
build_model(X, Y, 'dsl_model.sav')


X = np.array(crude_df['price'])
Y = hist_gas_df['R1'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=80)
X_train = X_train.reshape((-1,1))
X_test = X_test.reshape((-1,1))

lin_reg_model = pickle.load(open('Resources/data/reg_model.sav', 'rb'))

# R-squared (how much of the outcome is predicted correctly by the model) with training data
r_2_train = lin_reg_model.score(X_train, y_train)
print(r_2_train)

from matplotlib import pyplot as plt

predictions = lin_reg_model.predict(X_test)
r_2_test = lin_reg_model.score(X_test, y_test)
diffs = [0]*len(y_test)
for (pred, real, idx) in zip(predictions, y_test, range(len(y_test))):
    diffs[idx] = pred - real

# Plot of difference of real data from best fit line 
fig, ax = plt.subplots()
ax.plot(diffs)
fig.show()

# Visualizations
# Training Data Vis
plt.scatter(X_train, y_train, color = "mediumvioletred")
plt.plot(X_train, lin_reg_model.predict(X_train), color = "cornflowerblue")
plt.title("Model Prediction from Training Data")
plt.xlabel("Historical Crude Oil (USD/Barrel)")
plt.ylabel("Model Predicted Regular Gas (USD/Gallon)")
plt.text(20, 4.0, f"r^2 = {r_2_train:.3f}")
plt.savefig("Resources/images/training_fig.png")

# Test Data Vis
plt.scatter(X_test, y_test, color = "forestgreen")
plt.plot(X_test, lin_reg_model.predict(X_test), color = "coral")
plt.title("Model Prediction from Testing Data")
plt.xlabel("Historical Crude Oil (USD/Barrel)")
plt.ylabel("Model Predicted Regular Gas (USD/Gallon)")
plt.text(22, 3.2, f"r^2 = {r_2_test:.3f}")
plt.savefig("Resources/images/testing_fig.png")


# Twitter Sentiment Analysis (Natural Language Processing) Model 

### Description of Preliminary Data Preprocessing
#Using tweets from November 24, 2022 to December 5, 2022, the data are cleaned and subjectivity and polarity score columns are included.

### Description of Preliminary Feature Engineering and Preliminary Feature Selection
#To use natural language processing to characterize the sentiment of current tweets on gas prices (positive, neutral, or negative sentiment, ranging from -1.0 to 1.0).

# twitter_df = pd.read_csv('Resources/data/tweets_clean.csv')

# switchbag = twitter_df.values.reshape(43043, 1)
# twitter_df = pd.DataFrame(switchbag)

# # Adding column name to twitter_df
# twitter_df.columns = ['Tweet', 'sentiment']

# # Example with actual tweet
# example = TextBlob("rtr: U.S. seeks to limit flaring and methane leaks from public lands drilling - President Joe Biden's administration on Monday proposed rules aimed at limiting methane leaks from oil and gas drilling on public lands. By @ValerieVolco @nicholagroom https://t.co/Pi6vCf6RyC")

# for Tweet in twitter_df.columns:
#     a = TextBlob(Tweet)
#     twitter_df['sentiment'] = a.sentiment.polarity

# # Returning sentiment score of inputted tweet 
# def get_sentimental(target):
#     return TextBlob(target).sentiment.polarity

# # Converting all data to strings 
# def any2str(target):
#     if type(target) is not str:
#         return str(target)
#     else:
#         return target

# twitter_df['Tweet'] = twitter_df['Tweet'].apply(any2str)
# twitter_df['sentiment'] = twitter_df['Tweet'].apply(get_sentimental)

# # Visualizations 
# import matplotlib.pyplot as plt
# import math 
# import numpy as np
# import scipy.stats as stats

# def norm(x_min, x_max, avg, stdev):
#     x = np.arange(x_min, x_max, 0.01)
#     coeff = 1/(stdev*math.sqrt(2*math.pi))
#     expo = -0.5*(((x - avg)/ stdev)**2) 
#     return x, coeff*math.e**expo 

# # Data Vis of Sentiments
# sents = twitter_df['sentiment'].to_numpy()

# mu = twitter_df['sentiment'].mean()
# variance = twitter_df['sentiment'].var()
# sigma = twitter_df['sentiment'].std()
# bell_x, bell = norm(twitter_df['sentiment'].min(), twitter_df['sentiment'].max(), mu, sigma)
# #x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

# plt.hist(sents, color = "deeppink")
# plt.plot(bell_x, bell*(0.55*25000))
# plt.title("Sentiment Analysis Histogram")
# plt.xlabel("Sentiments of Gas Price Tweets (-1.0 to 1.0)")
# plt.ylabel("Count")

# plt.savefig("Resources/images/sentiment_hist.png")

### Explanation of Model Choice, including Limitations and Benefits
#This sentiment analysis NLP model provides a sentiment score for those who tweeted about gas and oil from November 24, 2022 to December 5, 2022. To understand the sentiment of current consumers, current data would need to be added. To include this sentiment analysis in the linear regression machine learning model, we would need more data, organized by month. Due to the limitations of the data and the data preprocessing of the linear regression model, there are only gas and diesel prices by month from January 1995 to January 2021 and February 2022 to December 2022. As a result, adding this sentiment analysis to the other model would only be adding two data points: November and December 2022 tweets.

### Train, Test, Split Description
#Since the tweet data are limited to November and December, they cannot be meaningfully added to the linear regression model, so there is no training set or testing set.

### Explanation of Changes in Model Choice & Description of How the Model Has Been Trained Thus Far and Any Additional Training
#Based on the formatting of the tweets and gas prices, the model went from attempting to predict current sentiment of gas prices to characterizing the sentiment from the last week in November to the first week in December.

### Description of Current Accuracy Score
#Not applicable.

### How the Model Addresses the Question/Problem the Team is Solving
#This sentiment analysis natural language processing model will compliment the linear regression model by showcasing general sentiment about current gas prices from tweets.