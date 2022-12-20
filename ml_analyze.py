import numpy as np
import pandas as pd
import pickle
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Linear Regression Machine Learning Model

## Description of Preliminary Feature Engineering and Preliminary Feature Selection
# To fit a linear regression model with crude oil and gas price data to establish and explore relationships between the data.

# Explanation of Model Choice, including Limitations and Benefits 
# This linear regression model is the best model because historically, crude oil prices and gas prices have a linear relationship, so there is no need to complicate the relationship with other, more complex models.

def predict_gas(crude_price):
    # Load trained linear regression models
    reg_model = pickle.load(open('Resources/data/reg_model.sav', 'rb'))
    mid_model = pickle.load(open('Resources/data/mid_model.sav', 'rb'))
    prm_model = pickle.load(open('Resources/data/prm_model.sav', 'rb'))
    dsl_model = pickle.load(open('Resources/data/dsl_model.sav', 'rb'))

    # Build dictionary with predicted gas prices
    prediction = {
    'Regular': reg_model.predict(np.array(crude_price).reshape(1, 1))[0],
    'MidGrade': mid_model.predict(np.array(crude_price).reshape(1, 1))[0],
    'Premium': prm_model.predict(np.array(crude_price).reshape(1, 1))[0],
    'Diesel': dsl_model.predict(np.array(crude_price).reshape(1, 1))[0]
    }

    # Return predicted gasoline price
    return prediction


# Twitter Sentiment Analysis (Natural Language Processing) Model 

### Description of Preliminary Data Preprocessing
#Using tweets from November 24, 2022 to December 5, 2022, the data are cleaned and subjectivity and polarity score columns are included.

### Description of Preliminary Feature Engineering and Preliminary Feature Selection
#To use natural language processing to characterize the sentiment of current tweets on gas prices (positive, neutral, or negative sentiment, ranging from -1.0 to 1.0).

def score_transform(score):
    if score < -0.1:
        return 'Negative'
    elif score > 0.1:
        return 'Positive'
    else:
        return 'Neutral'

def analyze_sentiment(tweets_df):
    # Calculate polarity score using textblob
    tweets_df['Sentiment_Score'] = tweets_df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Transform the polarity score into a categorical result
    tweets_df['Polarity'] = tweets_df['Sentiment_Score'].apply(score_transform)

    # Convert and return the finished dataframe
    tweets_df.index = tweets_df.index.map(str)
    return tweets_df.to_dict('records')