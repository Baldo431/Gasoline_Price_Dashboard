# Import Dependencies
from bs4 import BeautifulSoup,Comment
import numpy as np
import pandas as pd
import regex as re
import requests
from lxml.html.soupparser import fromstring
import prettify
import numbers
import htmltext
import json
import datetime


def gas_diesel_price():

    request_headers = {
        'accept':
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.8',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/601.3.9'
    }
    with requests.Session() as session:
        url = 'https://gasprices.aaa.com/'
        response = session.get(url, headers=request_headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    df = pd.DataFrame()

    html_data = soup.find(class_="table-mob")
    html_data2 = soup.find(class_="average-price")

    date = [span.get_text() for span in html_data2.find_all("span")]
    data = [td.get_text() for td in html_data.find_all("td")]

    CurrentDate = []
    GasPrice = []
    DieselPrice = []
    day = pd.to_datetime(date[0][11:])

    CurrentDate.append(day)
    GasPrice.append(round(float(data[1][1:]),2))
    DieselPrice.append(round(float(data[4][1:]),2))

    df['Date'] = CurrentDate
    df['Regular'] = GasPrice
    df['Diesel'] = DieselPrice

    gas = pd.read_csv("./gas-diesel-prices.csv")
    gas['Date'] = pd.to_datetime(gas['Date'])
    gas = pd.concat([gas, df], ignore_index = True)
    gas = gas.drop(columns=['Unnamed: 0'])
    gas.to_csv('gas-diesel-prices.csv')

    # Convert gas price data dataframe to dictionary.
    gas.index = gas.index.map(str)
    gas_price_dict = gas.to_dict()

    return gas_price_dict
