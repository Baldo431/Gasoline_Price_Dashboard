# Import Splinter and BeautifulSoup
from splinter import Browser
from bs4 import BeautifulSoup as soup
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import datetime as dt

def scrape_all():
    # Initiate headless driver for deployment
    executable_path = {'executable_path': ChromeDriverManager().install()}
    browser = Browser('chrome', **executable_path, headless=True)

    # Run scrape functions
    gas_prices = current_pricing(browser)

    # Store results in dictionary
    data = {
        "todays_prices": gas_prices,
        "last_modified": dt.datetime.now()
    }

    # End Splinter session
    browser.quit()

    return data

# Retrieve current gas prices from AAA.com
def current_pricing(browser):
    # Add try/except for error handling
    try:
        # Visit the AAA site
        url = 'https://gasprices.aaa.com/'
        browser.visit(url)

        # Delay for loading the page
        browser.is_element_present_by_css('div.list_text', wait_time=1)
    except BaseException:
        return None

    prices_data = []
    fuel_types = []

    # Convert the base url visited to html
    prices_soup = soup(browser.html, 'html.parser')

    #Search for the objects that contain the fuel types and the fuel prices.
    table_elem = prices_soup.find('table', class_='table-mob')
    title_elem = table_elem.find_all('th')
    prices_elem = table_elem.find('tbody').find_all('tr')[0].find_all('td')

    # Extract the text for fuel types
    for item in title_elem:
        if item.get_text() != '':
            fuel_types.append(item.get_text())
            
    # Extract the text for fuel prices
    for item in prices_elem:
        if item.get_text()[0] == '$':
            prices_data.append(item.get_text())

    # Combine the two lists into a dictionary.
    prices_dict = dict(zip(fuel_types, prices_data))
    return prices_dict


def historical_pricing():
    # Import static csv data into dataframe
    crude_hist_df = pd.read_csv('./Resources/crude_price_history.csv')
    gas_hist_df = pd.read_csv('./Resources/gas_price_history.csv', header=2)

    # Clean gas dataframe
    gas_hist_df['Date'] = pd.to_datetime(gas_hist_df['Date'])
    gas_hist_df['Date'].dropna()
    gas_hist_df.drop(['Unnamed: 16', 'Weekly U.S. No 2 Diesel Low Sulfur (15-500 ppm) Retail Prices  (Dollars per Gallon)'], axis=1, inplace=True)

    # Filter gas dataframe to only include data from last year.
    now = datetime.now()
    monday = now - timedelta(days = now.weekday()) - relativedelta(years=1)
    gas_hist_filtered = gas_hist_df[gas_hist_df['Date'] >= monday]

    # Convert gas dataframe to dictionary in order to import to mongodb.
    gas_hist_filtered.reset_index(inplace=True)
    gas_hist_filtered.drop(['index'], axis=1, inplace=True)
    gas_hist_filtered.index = gas_hist_filtered.index.map(str)
    gas_dict = gas_hist_filtered.to_dict()

    # Convert crude oil dataframe to dictionary in order to import to mongodb.
    crude_hist_df.index = crude_hist_df.index.map(str)
    crude_dict = crude_hist_df.to_dict()

    return gas_dict, crude_dict