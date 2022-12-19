# see gas_oil_requirements.txt 

# Gas & Oil Linear Regression Model
import os
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Changing Directory
os.chdir('/Users/annettedblackburn/Desktop/Data_Analytics_Bootcamp/Module 20 - Final (Group) Project')

os.listdir(os.curdir)

## Description of Preliminary Data Preprocessing
# Starting with crude oil data (1983 to present) and gas and diesel price data (1995 to 2021), the data are cleaned to fit into a linear regression machine learning model.
# Below the data are loaded into two dataframes (gas_df and crude_df). The cleaned gas price dataframe includes all formulations of retail gasoline and diesel prices in a MM/DD/YYYY format with samples from each month starting in January 1995 to January 2021. The cleaned crude oil dataframe is in the same format as the cleaned gas price dataframe: MM/DD/YYYY format with monthly samples from January 1995 to January 2021.
## Data Examining & Cleaning 

hist_gas_df = pd.read_csv("PET_PRI_GND_DCUS_NUS_W.csv")

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

# Dropping conventional and reformulated retail gas prices as well as all, mid-grade, and premium
hist_gas_df['date'] = hist_gas_df['Date']
hist_gas_df = hist_gas_df[['date', 'R1', 'D1']]

hist_crude_df = pd.read_csv("crude-oil-price.csv")

# Dropping percent change and change of crude oil prices
hist_crude_df = hist_crude_df[['date', 'price']]

# Formatting crude_df to match hist_gas_df date format
from datetime import date
def fix_date(bad_date):
    bad_date = str(bad_date)
    year, month, day = bad_date.split('-')
    day = day[0:2]
    return date(int(year), int(month), int(day))

hist_crude_df['date'] = hist_crude_df['date'].apply(fix_date) 

# Dropping 1983-1994 from hist_crude_df
cuttoff_date = date(1994, 12, 31)
hist_crude_df = hist_crude_df.loc[hist_crude_df['date'] > cuttoff_date]

# Lining up months and years in crude_df and gas_df
def date2str(dt):
    return dt.strftime('%m/%Y')

def drop_month(dt):
    month, day, year = dt.split('/')
    return f"{month}/{year}"

hist_crude_df['date'] = hist_crude_df['date'].apply(date2str)
hist_gas_df['date'] = hist_gas_df['date'].apply(drop_month)

# Omit 2022 data 
hist_crude_df = hist_crude_df.loc[0:454]

# Current Crude Oil Price
current_crude_df = pd.read_csv("current-oil-prices.csv")
current_crude_df = current_crude_df[['Date', 'Crude Closing']]

# Rename columns so current_crude_df and hist_crude_df line up
current_crude_df = current_crude_df.rename(columns={'Date': 'date', 'Crude Closing': 'price'})
def ref_date(date):
    date = str(date)
    y, m, d = date.split('-')
    return f"{m}/{y}"
current_crude_df['date'] = current_crude_df['date'].apply(ref_date)
current_crude_df.head()

# Keep only first Month/Year instance for current_crude_df
seen_dates = set()
indices_to_remove = []
for idx, row in current_crude_df.iterrows():
    _date = row['Date']
    if _date in seen_dates:
        indices_to_remove.append(idx)
    else:
        seen_dates.add(_date)
current_crude_df.drop(indices_to_remove, inplace=True)

# Appending hist_crude_df and current_crude_df
crude_df = pd.concat([hist_crude_df, current_crude_df], axis=0)

# Rename columns so current_crude_df and hist_crude_df line up
current_crude_df = current_crude_df.rename(columns={'Date': 'date', 'Crude Closing': 'price'})

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

# Current Gas Prices DF Data Cleaning
current_gas_df = pd.read_csv("gas-diesel-prices.csv")

# Lining up months and years in current_gas_df to match crude_df and hist_gas_df
def date2str(dt):
    return dt.strftime('%m/%Y')

def drop_month(dt):
    year, month, day = dt.split('-')
    return f"{month}/{year}"
current_gas_df['date'] = current_gas_df['Date'].apply(drop_month)

# Keep only first Month/Year instance for current_gas_df
seen_dates = set()
indices_to_remove = []
for idx, row in current_gas_df.iterrows():
    _date = row['date']
    if _date in seen_dates:
        indices_to_remove.append(idx)
    else:
        seen_dates.add(_date)
current_gas_df.drop(indices_to_remove, inplace=True)

current_gas_df = current_gas_df[['date', 'Regular', 'Diesel']]

# Appending hist_gas_df and current_gas_df
gas_df = hist_gas_df.append(current_gas_df)

# Appending hist_crude_df and current_crude_df
crude_df = pd.concat([hist_crude_df, current_crude_df], axis=0)

# Linear Regression Machine Learning Model

## Description of Preliminary Feature Engineering and Preliminary Feature Selection
# To fit a linear regression model with crude oil and gas price data to establish and explore relationships between the data.

# Explanation of Model Choice, including Limitations and Benefits 
# This linear regression model is the best model because historically, crude oil prices and gas prices have a linear relationship, so there is no need to complicate the relationship with other, more complex models.

# Gas Linear Reg Model 
# X = crude oil price [=] 1/barrel
# Y = gas price [=] 1/gallon
import numpy as np

X = np.array(crude_df['price'])
Y = gas_df['Regular'].to_numpy()
f"X.shape = {X.shape}, Y.shape = {Y.shape}"

## Train, Test, Split Description 
# The data are split into 80% training and 20% testing.

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

X_train = X_train.reshape((-1,1))
X_test = X_test.reshape((-1,1))

gas_lin_reg_model = LinearRegression()
gas_lin_reg_model.fit(X_train, Y_train)

# R-squared (how much of the outcome is predicted correctly by the model) with training data
r_2_train = gas_lin_reg_model.score(X_train, Y_train)
r_2_train

from matplotlib import pyplot as plt

predictions = gas_lin_reg_model.predict(X_test)
r_2_test = gas_lin_reg_model.score(X_test, Y_test)
diffs = [0]*len(Y_test)
for (pred, real, idx) in zip(predictions, Y_test, range(len(Y_test))):
    diffs[idx] = pred - real

# Plot of difference of real data from best fit line 
fig, ax = plt.subplots()
ax.plot(diffs)
fig.show()

# R-squared (how much of the outcome is predicted correctly by the model) with test data
r_2_test

# Visualizations for Gas Lin Reg Model 
# Training Data Vis
plt.scatter(X_train, Y_train, color = "forestgreen")
plt.plot(X_train, gas_lin_reg_model.predict(X_train), color = "coral")
plt.title("Model Prediction from Training Data")
plt.xlabel("Historical Crude Oil (USD/Barrel)")
plt.ylabel("Model Predicted Regular Gas (USD/Gallon)")
plt.text(20, 4.0, f"r^2 = {r_2_train:.3f}")
plt.savefig("gas_training_fig.png")
# Test Data Vis
plt.scatter(X_test, Y_test, color = "forestgreen")
plt.plot(X_test, gas_lin_reg_model.predict(X_test), color = "coral")
plt.title("Model Prediction from Testing Data")
plt.xlabel("Historical Crude Oil (USD/Barrel)")
plt.ylabel("Model Predicted Regular Gas (USD/Gallon)")
plt.text(22, 3.2, f"r^2 = {r_2_test:.3f}")
plt.savefig("gas_testing_fig.png")

# Diesel Lin Reg Model
# X = crude oil price [=] 1/barrel
# Y = gas price [=] 1/gallon
import numpy as np

X = np.array(crude_df['price'])
Y = gas_df['Diesel'].to_numpy()
f"X.shape = {X.shape}, Y.shape = {Y.shape}"

#### Train, Test, Split Description 
# The data are split into 80% training and 20% testing.

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=80)

X_train = X_train.reshape((-1,1))
X_test = X_test.reshape((-1,1))

X_train.shape

diesel_lin_reg_model = LinearRegression()
diesel_lin_reg_model.fit(X_train, Y_train)

# R-squared (how much of the outcome is predicted correctly by the model) with training data
r_2_train = diesel_lin_reg_model.score(X_train, Y_train)
r_2_train

from matplotlib import pyplot as plt

predictions = diesel_lin_reg_model.predict(X_test)
r_2_test = diesel_lin_reg_model.score(X_test, Y_test)
diffs = [0]*len(Y_test)
for (pred, real, idx) in zip(predictions, Y_test, range(len(Y_test))):
    diffs[idx] = pred - real

# Plot of difference of real data from best fit line 
fig, ax = plt.subplots()
ax.plot(diffs)
fig.show()

# R-squared (how much of the outcome is predicted correctly by the model) with test data
r_2_test

# Visualizations for Diesel Lin Reg Model
# Training Data Vis
plt.scatter(X_train, Y_train, color = "mediumvioletred")
plt.plot(X_train, diesel_lin_reg_model.predict(X_train), color = "cornflowerblue")
plt.title("Model Prediction from Training Data")
plt.xlabel("Historical Crude Oil (USD/Barrel)")
plt.ylabel("Model Predicted Diesel (USD/Gallon)")
plt.text(20, 4.0, f"r^2 = {r_2_train:.3f}")
plt.savefig("diesel_training_fig.png")
# Test Data Vis
plt.scatter(X_test, Y_test, color = "mediumvioletred")
plt.plot(X_test, diesel_lin_reg_model.predict(X_test), color = "cornflowerblue")
plt.title("Model Prediction from Testing Data")
plt.xlabel("Historical Crude Oil (USD/Barrel)")
plt.ylabel("Model Predicted Diesel (USD/Gallon)")
plt.text(22, 3.2, f"r^2 = {r_2_test:.3f}")
plt.savefig("diesel_testing_fig.png")

# Dictionary Format for Website
# Function that takes in Current Crude Oil Price and returns predicted Gas Price in dictionary format
def predict_price(crude_price):
    crude_price = np.array([crude_price])
    crude_price = np.expand_dims(crude_price, axis=1)
    gas_prediction = gas_lin_reg_model.predict(crude_price)
    diesel_prediction = diesel_lin_reg_model.predict(crude_price)
    return {"Current Closing Price": crude_price[0][0], "Predicted Regular Gas Price": gas_prediction[0], "Predicted Diesel Price": diesel_prediction[0]}

# Example of 12/19/22 crude oil price ($75.65)
predict_price(75.65)

# Results: {'Current Closing Price': 75.65, 'Predicted Regular Gas Price': 2.9166473070570174,'Predicted Diesel Price': 3.2435708724468726}
# 12/19/22 current national average regular gas price ($3.142) and diesel price ($4.75)


# Twitter Sentiment Analysis (Natural Language Processing) Model 
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Changing Directory
os.chdir('/Users/annettedblackburn/Desktop/Data_Analytics_Bootcamp/Module 20 - Final (Group) Project')

os.listdir(os.curdir)

### Description of Preliminary Data Preprocessing
#Using tweets from November 24, 2022 to December 5, 2022, the data are cleaned and subjectivity and polarity score columns are included.

### Description of Preliminary Feature Engineering and Preliminary Feature Selection
#To use natural language processing to characterize the sentiment of current tweets on gas prices (positive, neutral, or negative sentiment, ranging from -1.0 to 1.0).

twitter_mess = pd.read_csv("tweets_clean.csv")
twitter_mess.head()

twitter_dict = {'date': [], 'tweet': []}
for column_name, column_content in twitter_mess.items():
    if column_name=="Unnamed: 0": continue
    current_date = str(column_name)
    for current_tweet in column_content: 
        twitter_dict['date'].append(current_date)
        twitter_dict['tweet'].append(current_tweet)

twitter_df = pd.DataFrame(twitter_dict)
twitter_df

# Example with actual tweet
example = TextBlob("rtr: U.S. seeks to limit flaring and methane leaks from public lands drilling - President Joe Biden's administration on Monday proposed rules aimed at limiting methane leaks from oil and gas drilling on public lands. By @ValerieVolco @nicholagroom https://t.co/Pi6vCf6RyC")

for Tweet in twitter_df.columns:
    a = TextBlob(Tweet)
    twitter_df['sentiment'] = a.sentiment.polarity

sentiments = []
for Tweet in twitter_df['tweet'].to_list():
    a = TextBlob(Tweet)
    sentiments.append(a.sentiment.polarity)

twitter_df['sentiment'] = sentiments

# Returning sentiment score of inputted tweet 
def get_sentimental(target):
    return TextBlob(target).sentiment.polarity

# Converting all data to strings 
def any2str(target):
    if type(target) is not str:
        return str(target)
    else:
        return target

twitter_df['tweet'] = twitter_df['tweet'].apply(any2str)
twitter_df['sentiment'] = twitter_df['tweet'].apply(get_sentimental)

# Dictionary Format for Website 
twitter_dict = twitter_df.to_dict()

# Visualizations 
import matplotlib.pyplot as plt
import math 
import numpy as np
import scipy.stats as stats

def norm(x_min, x_max, avg, stdev):
    x = np.arange(x_min, x_max, 0.01)
    coeff = 1/(stdev*math.sqrt(2*math.pi))
    expo = -0.5*(((x - avg)/ stdev)**2) 
    return x, coeff*math.e**expo 

# Data Vis of Sentiments
sents = twitter_df['sentiment'].to_numpy()

mu = twitter_df['sentiment'].mean()
variance = twitter_df['sentiment'].var()
sigma = twitter_df['sentiment'].std()
bell_x, bell = norm(twitter_df['sentiment'].min(), twitter_df['sentiment'].max(), mu, sigma)
#x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

plt.hist(sents, color = "deeppink")
plt.plot(bell_x, bell*(0.55*25000))
plt.title("Sentiment Analysis Histogram")
plt.xlabel("Sentiments of Gas Price Tweets (-1.0 to 1.0)")
plt.ylabel("Count")

plt.savefig("sentiment_hist.png")

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