from scraping import gas_diesel_price
from twitter_data import twitter
from ml_analyze import predict_gas, analyze_sentiment
from pymongo import MongoClient


if __name__ == "__main__":

    # Set up mongo connection
    client = MongoClient("mongodb://localhost:27017/") 
    db = client.gas_app

    # Run functions to gather data
    latest_crude_price, current_price, gas_crude_hist = gas_diesel_price()
    predicted_price = predict_gas(latest_crude_price)
    tweet_count, tweet_df = twitter()
    tweet_data = analyze_sentiment(tweet_df)

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
    col.insert_many(tweet_data)
