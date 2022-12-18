# Import Dependencies
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from datetime import timedelta
import tweepy

from config import consumer_key
from config import consumer_secret
from config import access_token
from config import access_token_secret
from config import token


def twitter():

    # RECENT TWEET COUNT ======================================================

    # Twitter API Bearer Token.
    token = token 

    # Query Tweets that include gas and oil.
    query = "gas oil -is:retweet"
    client = tweepy.Client(bearer_token=token)

    # Get count of recent tweets related to query in the last day.
    counts = client.get_recent_tweets_count(query=query, 
                                            granularity='day')
    Lst = []

    for i in counts.data:
        Dict = {}
        Dict['Day'] = str(i["start"][0:10])
        Dict['Tweet Count'] = i["tweet_count"]
        Lst.append(Dict)

    # Change to dataframe.
    df = pd.DataFrame(Lst)
    df = pd.DataFrame.from_dict(Lst)
    df = pd.DataFrame.from_records(Lst)
    df = df.rename(columns={"Day": "Date"})
    df['Date'] = pd.to_datetime(df['Date'])

    # Locate today's count.
    update_df = df.iloc[-2:-1]

    # Open previous data.
    tweet_count = pd.read_csv("./tweet_count.csv")
    tweet_count = tweet_count.rename(columns={"d": "Date"})
    tweet_count['Date'] = pd.to_datetime(tweet_count['Date'])

    # Add today's data to previous data.
    tweet_count = pd.concat([tweet_count, update_df], ignore_index = True)

    # Save updated data to csv.
    tweet_count = tweet_count.drop(columns=['Unnamed: 0'])
    tweet_count.to_csv('tweet_count.csv')

    # Convert tweet counts dataframe to dictionary.
    tweet_count.index = tweet_count.index.map(str)
    tweet_dict = tweet_count.to_dict()


    # RECENT TWEETS ======================================================

    # Get today's date and tomorrow's date to run time frame for today.
    day1 = date.today()
    day2 = day1 + timedelta(days = 1)
    day1 = str(day1)
    day2 = str(day2)

    # Scrape Twitter for all tweets related to query (this might take a minute).
    start = day1 + 'T00:00:00.000Z'
    end = day2 + 'T00:00:00.00Z'

    tweets_list = []

    tweets = tweepy.Paginator(client.search_recent_tweets, query=query,
                            tweet_fields=['context_annotations', 'created_at'],
                            start_time=start, 
                            end_time=end,
                            max_results=100).flatten(limit=10000)

    for tweet in tweets:
        tweets_list.append(tweet.text)

    # Save all tweets to dataframe and clean data.
    tweet_df = pd.DataFrame(tweets_list)
    tweet_df.columns = [day1]
    tweet_df = tweet_df.drop_duplicates()

    # Save today's tweets to csv file.
    tweet_df.to_csv(day1 + ".csv", index=True) 

    # Get previous tweets data.
    all_tweets = pd.read_csv("./tweets.csv")

    # Merge today's tweet data with old data.
    all_tweets = all_tweets.join(tweet_df, how='outer')

    # Clean data and save it to csv files. 
    all_tweets = all_tweets.drop(columns=['Unnamed: 0'])
    all_tweets.to_csv('tweets.csv') # (tweets.csv = raw tweets)
    all_tweets = all_tweets.dropna()
    all_tweets.to_csv('tweets_clean.csv') # (tweets_clean.csv = cleaned tweets)

    # Convert all tweet dataframe to dictionary.
    all_tweets.index = all_tweets.index.map(str)
    all_tweets_dict = all_tweets.to_dict()

    return tweet_dict, all_tweets_dict
