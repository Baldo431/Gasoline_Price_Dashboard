# Import Dependencies
import os
import tweepy
import pandas as pd
from datetime import date
from datetime import timedelta
from splinter import Browser
from bs4 import BeautifulSoup as soup
from webdriver_manager.chrome import ChromeDriverManager


from config import consumer_key
from config import consumer_secret
from config import access_token
from config import access_token_secret
from config import token

def get_embed_code(tweet_id, browser):
    try:
        url='https://publish.twitter.com/#'
        browser.visit(url)

        # Delay for loading the page
        browser.is_element_present_by_css('div.list_text', wait_time=1)
    except BaseException:
        return None
    
    #Find query form and fill it in with twe
    query_form = browser.find_by_id('configuration-query').first
    query_form.fill('https://twitter.com/twitter/statuses/'+str(tweet_id))
    
    browser.find_by_tag('button').first.click()
    return browser.find_by_tag('code').first.text

def twitter():

    # RECENT TWEET COUNT ======================================================

    # Twitter API Bearer Token.
    api_token = token 

    # Query Tweets that include gas and oil.
    query = "gas oil -is:retweet"
    client = tweepy.Client(bearer_token=api_token)

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
    tweet_count = pd.read_csv("Resources/data/tweet_count.csv")
    tweet_count = tweet_count.rename(columns={"d": "Date"})
    tweet_count['Date'] = pd.to_datetime(tweet_count['Date'])

    # Add today's data to previous data.
    tweet_count = pd.concat([tweet_count, update_df], ignore_index = True)

    # Save updated data to csv.
    tweet_count = tweet_count.drop(columns=['Unnamed: 0'])
    tweet_count.drop_duplicates(subset=["Date"],inplace=True)
    tweet_count.to_csv('Resources/data/tweet_count.csv')

    # Convert tweet counts dataframe to dictionary.
    tweet_count.index = tweet_count.index.map(str)
    tweet_count_dict = tweet_count.to_dict()


    # RECENT TWEETS ======================================================

    # Get today's date and tomorrow's date to run time frame for today.
    day1 = date.today()
    day2 = day1 - timedelta(days = 1)
    day1 = str(day1)
    day2 = str(day2)

    # Scrape Twitter for all tweets related to query (this might take a minute).
    start = day2 + 'T00:00:00.000Z'
    end = day1 + 'T00:00:00.00Z'

    tweets_list = []

    tweets = tweepy.Paginator(client.search_recent_tweets, query=query,
                            tweet_fields=['context_annotations', 'created_at'],
                            start_time=start, 
                            end_time=end,
                            max_results=100).flatten(limit=100)

    for tweet in tweets:
        tweets_list.append(tweet)
                            
    recent_tweets_df = pd.DataFrame({'Date': pd.Series(dtype='str'),
        'Tweet_ID': pd.Series(dtype='str'),
        'Text': pd.Series(dtype='str')})

    for item in tweets_list:
        recent_tweets_df = recent_tweets_df.append({
            'Date': item.created_at,
            'Tweet_ID': item.id,
            'Text': item.text
        }, ignore_index = True)

    # Initiate headless driver for deployment
    executable_path = {'executable_path': ChromeDriverManager().install()}
    browser = Browser('chrome', **executable_path, headless=True)

    # Create a new column and add the scraped code
    print("Starting Process: Scraping Tweet Embedding Code (Please Note that this may take a while.")
    recent_tweets_df['Embed_Code'] = recent_tweets_df['Tweet_ID'].apply(get_embed_code, browser=browser)
    browser.quit()
    print("Finished Process!")

    return tweet_count_dict, recent_tweets_df

    # for tweet in tweets:
    #     tweets_list.append(tweet.text)

    # # Save all tweets to dataframe and clean data.
    # tweet_df = pd.DataFrame(tweets_list)
    # tweet_df.columns = [day1]
    # tweet_df = tweet_df.drop_duplicates()

    # # Save today's tweets to csv file.
    # tweet_df.to_csv(day1 + ".csv", index=True) 

    # # Get previous tweets data.
    # all_tweets = pd.read_csv("./tweets.csv")

    # # Merge today's tweet data with old data.
    # all_tweets = all_tweets.join(tweet_df, how='outer')

    # # Clean data and save it to csv files. 
    # all_tweets = all_tweets.drop(columns=['Unnamed: 0'])
    # all_tweets.to_csv('tweets.csv') # (tweets.csv = raw tweets)
    # all_tweets = all_tweets.dropna()
    # all_tweets.to_csv('tweets_clean.csv') # (tweets_clean.csv = cleaned tweets)


    # ts_df = pd.DataFrame()
    # ts_list = []
    # append_data = []

    # for (columnName, columnData) in all_tweets.iteritems():
    #     case = {columnName : columnData.values}
    #     ts_list.append(case)

    # for i in range(len(ts_list)):
    #     ts_series = pd.Series(ts_list[i], name='Raw_Text').rename_axis('Date').explode().reset_index()
    #     append_data.append(ts_series)
    # ts_df = pd.concat(append_data)
    # ts_df = ts_df.reset_index()
    # ts_df = ts_df.drop(columns=['index'])

    # # Save transposed tweets to csv file.
    # ts_df.to_csv('transposed_tweets.csv')

    # ts_df.index = ts_df.index.map(str)
    # transposed_tweets_dict = ts_df.to_dict()

    # return tweet_count_dict, transposed_tweets_dict