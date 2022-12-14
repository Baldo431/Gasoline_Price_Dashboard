# Gas & Oil Linear Regression Model
import os
import numpy as np
import pandas as pd
import spacy
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

gas_df = pd.read_csv("PET_PRI_GND_DCUS_NUS_W.csv")
gas_df.head()

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

# Dropping conventional and reformulated retail gas prices
gas_df['date'] = gas_df['Date']
gas_df = gas_df[['date', 'A1', 'R1', 'M1', 'P1', 'D1']]
gas_df.head(10)

crude_df = pd.read_csv("crude-oil-price.csv")
crude_df.head()

# Dropping percent change and change of crude oil prices
crude_df = crude_df[['date', 'price']]
crude_df.head()

# Formatting crude_df to match gas_df date format
from datetime import date
def fix_date(bad_date):
    bad_date = str(bad_date)
    year, month, day = bad_date.split('-')
    day = day[0:2]
    return date(int(year), int(month), int(day))

crude_df['date'] = crude_df['date'].apply(fix_date) 
crude_df.head()

# Dropping 1983-1994 from crude_df
cuttoff_date = date(1994, 12, 31)
crude_df = crude_df.loc[crude_df['date'] > cuttoff_date]
crude_df.head(10)

# Lining up months and years in crude_df and gas_df
def date2str(dt):
    return dt.strftime('%m/%Y')

def drop_month(dt):
    month, day, year = dt.split('/')
    return f"{month}/{year}"

crude_df['date'] = crude_df['date'].apply(date2str)
gas_df['date'] = gas_df['date'].apply(drop_month)

# Price per barrel 
crude_df.head()

crude_df.tail(25)

# Omit 2022 data 
crude_df = crude_df.loc[0:454]
crude_df.tail(10)

# Price per gallon 
gas_df.head()

gas_df.tail(15)

# Keep only first Month/Year instance for gas_df
seen_dates = set()
indices_to_remove = []
for idx, row in gas_df.iterrows():
    _date = row['date']
    if _date in seen_dates:
        indices_to_remove.append(idx)
    else:
        seen_dates.add(_date)
gas_df.drop(indices_to_remove, inplace=True)
gas_df.head()

# Linear Regression Machine Learning Model

## Description of Preliminary Feature Engineering and Preliminary Feature Selection
# To fit a linear regression model with crude oil and gas price data to establish and explore relationships between the data.

# Explanation of Model Choice, including Limitations and Benefits 
# This linear regression model is the best model because historically, crude oil prices and gas prices have a linear relationship, so there is no need to complicate the relationship with other, more complex models.

# X = crude oil price [=] 1/barrel
# Y = gas price [=] 1/gallon
import numpy as np

X = np.array(crude_df['price'].values.tolist())
y = np.array(gas_df[['A1', 'R1', 'M1', 'P1', 'D1']].values.tolist())

## Train, Test, Split Description 
# The data are split into 80% training and 20% testing.

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

X_train = X_train.reshape((-1,1))
X_test = X_test.reshape((-1,1))

lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)

# R-squared (how much of the outcome is predicted correctly by the model) with training data
r_2_train = lin_reg_model.score(X_train, y_train)
r_2_train

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

# R-squared (how much of the outcome is predicted correctly by the model) with test data
r_2_test


# Twitter Sentiment Analysis (Natural Language Processing) Model 
import os

import pandas as pd

import numpy as np 
from pathlib import Path
from collections import Counter

from tqdm import tqdm
import json
import pandas as pd

import re

import matplotlib.pyplot as plt
import seaborn as sns

import spacy
nlp = spacy.load('en_core_web_sm')

from spacytextblob.spacytextblob import SpacyTextBlob
nlp.add_pipe('spacytextblob')

from string import punctuation as PUNCTUATION
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

# Changing Directory
os.chdir('/Users/annettedblackburn/Desktop/Data_Analytics_Bootcamp/Module 20 - Final (Group) Project')

os.listdir(os.curdir)

twitter_df = pd.read_csv('tweets_clean.csv')
twitter_df

switchbag = twitter_df.values.reshape(43043, 1)
twitter_df = pd.DataFrame(switchbag)
twitter_df

twitter_df = pd.read_csv('tweets_clean.csv')
polarity_scores = []
subjectivity_scores = []
tweets = []

with open('tweets_clean.csv', 'r') as f:
    for line in tqdm(f, total=10000):
        tweet_dict = line.split(",")
        tweets = twitter_df
        tweets_doc = nlp(tweet_dict)
        polarity_scores.append(tweets._.blob.polarity)
        subjectivity_scores.append(tweets._.blob.subjectivity)
        tweets.append(tweets)

twitter_df['polarity_score'] = polarity_scores
twitter_df['subjectivity_score'] = subjectivity_scores
twitter_df['tweets'] = tweets

twitter_df.tail()
# current twitter_df format: 3310, 13 (rows, columns)
# new twitter_df format: 43043, 1 (rows, columns)

switchbag = twitter_df.values.reshape(43043, 1)
twitter_df = pd.DataFrame(switchbag)

twitter_df.head()

# Adding column name to tweets
twitter_df = twitter_df.rename(columns = {'0':'Tweets'})
twitter_df.head()

twitter_df = pd.DataFrame(columns=['polarity_score', 'subjectivity_score'])
twitter_df

# Adding columns 
polarity_scores = []
subjectivity_scores = []
tweets = []

for tweet in tweets:
    doc = nlp(tweet)
    for token in doc:
        print

with open('tweets_clean.csv', 'r') as f:
    for line in tqdm(f, total=10000):
        #tweet_dict, tweets, tweets_doc = 
       #tweet_dict = json.loads(line)
        tweets = tweet_dict['text']
        tweets_doc = nlp(tweets)
        # 4. polarity score for each tweet (emotions expressed)
        polarity_scores.append(tweets_doc._.blob.polarity)
        # 5. subjectivity score for each tweet (personal feelings, views, beliefs expressed in the tweet)
        subjectivity_scores.append(tweets_doc._.blob.subjectivity)
        tweets.append(tweets)

twitter_df['polarity_score'] = polarity_scores
twitter_df['subjectivity_score'] = subjectivity_scores
twitter_df['tweets'] = tweets

twitter_df.head()

# Drop NAs
twitter_df.dropna()

## Using spacy's English language model, individual tweets are tokenized and the following are collected:
#1. total number of words
#2. total number of punctuation marks
#3. total number of words that are not stop words
#4. polarity score (emotions expressed in the tweet)
#5. subjectivity score (personal feelings, views, or beliefs expressed in the tweet)

# 1. total number of words
def count_words(text):
    words = text.split()
    return len(words)

twitter_df['word_count'] = twitter_df["review"].apply(count_words)
twitter_df.head()

# 2. total number of punctuation marks for each tweet
def count_punct_marks(text):
    char_count = 0
    for cha in text:
        if cha in PUNCTUATION:
            char_count+=1
    return char_count

twitter_df['punct_count'] = twitter_df["review"].apply(count_punct_marks)
twitter_df.head()

# 3. total number of words that are not stop words for each tweet
def count_non_stop_words(text):
    non_stop_words = pd.Series(text.split())
    non_stop_words.str.strip(PUNCTUATION)

    drop_list = []
    
    for idx, word in enumerate(non_stop_words):
        if word in STOP_WORDS:
            drop_list.append(idx)
    non_stop_words.drop(index=drop_list, inplace=True)
    return len(non_stop_words)


twitter_df['non_stop_words'] = twitter_df["review"].apply(count_non_stop_words)
twitter_df.head()

## Description of Preliminary Data Preprocessing
## Data Examining & Cleaning 
# Sentiment Analysis Natural Language Processing Machine Learning Model
## Description of Preliminary Feature Engineering and Preliminary Feature Selection
# Explanation of Model Choice, including Limitations and Benefits 
