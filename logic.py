import scraping
from pymongo import MongoClient

if __name__ == "__main__":
    # Set up mongo connection
    client = MongoClient("mongodb://localhost:27017/") 
    db = client.gas_app

    # Run functions to gather data
    current_price = scraping.scrape_all()
    gas_hist, crude_hist = scraping.historical_pricing()

    # Push data to mongodb
    col = db.current_pricing
    col.update_one({}, {"$set": current_price}, upsert=True)

    col = db.gas_history
    col.update_one({}, {"$set": gas_hist}, upsert=True)

    col = db.crude_history
    col.update_one({}, {"$set": crude_hist}, upsert=True)