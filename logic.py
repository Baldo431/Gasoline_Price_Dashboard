import scraping
import twitter_data
import ml_analyze
from pymongo import MongoClient

if __name__ == "__main__":
    # Set up mongo connection
    client = MongoClient("mongodb://localhost:27017/") 
    db = client.gas_app

    # Run functions to gather data
    latest_crude_price, current_price, gas_crude_hist = scraping.gas_diesel_price()
    predicted_price = ml_analyze.predict_gas(latest_crude_price)
    tweet_count, tweet_df = twitter_data.twitter()
    tweet_data = ml_analyze.analyze_sentiment(tweet_df)

    # Push data to mongodb
    col = db.current_pricing
    col.update_one({}, {"$set": current_price}, upsert=True)

    col = db.predicted_pricing
    col.update_one({}, {"$set": predicted_price}, upsert=True)

    col = db.gas_crude_history
    col.update_one({}, {"$set": gas_crude_hist}, upsert=True)

    col = db.tweet_counts
    col.update_one({}, {"$set": tweet_count}, upsert=True)

    col = db.tweet_data
    col.update_one({}, {"$set": tweet_data}, upsert=True)