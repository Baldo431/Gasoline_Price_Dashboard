# Import Dependencies
from bs4 import BeautifulSoup,Comment
import pandas as pd
import requests


def gas_diesel_price():

    request_headers = {
        'accept':
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.8',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/601.3.9'
    }

    # Gas Prices

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
    RegularPrice = []
    MidGradePrice = []
    PremiumPrice = []
    DieselPrice = []
    day = pd.to_datetime(date[0][11:])

    CurrentDate.append(day)
    RegularPrice.append(round(float(data[1][1:]),2))
    MidGradePrice.append(round(float(data[2][1:]),2))
    PremiumPrice.append(round(float(data[3][1:]),2))
    DieselPrice.append(round(float(data[4][1:]),2))

    df['Date'] = CurrentDate
    df['Regular'] = RegularPrice
    df['MidGrade'] = MidGradePrice
    df['Premium'] = PremiumPrice
    df['Diesel'] = DieselPrice

    gas = pd.read_csv("Resources/data/gas-diesel-prices.csv")
    gas['Date'] = pd.to_datetime(gas['Date'])
    gas = pd.concat([gas, df], ignore_index = True)
    gas = gas.drop(columns=['Unnamed: 0'])
    gas = gas.drop_duplicates()
    gas.to_csv('Resources/data/gas-diesel-prices.csv')

    # Oil Prices

    with requests.Session() as session:
        url = 'https://www.marketwatch.com/investing/future/cl.1'
        response2 = session.get(url, headers=request_headers)

    soup2 = BeautifulSoup(response2.content, 'html.parser')
    df2 = pd.DataFrame()

    html_data3 = soup2.find('th', attrs={'class':'table__heading'})
    html_data4 = soup2.find('td', attrs={'class':'table__cell u-semi'})

    html_data3 = html_data3.text.replace('\n    {\n        ','{')
    html_data3 = html_data3.replace('\n    }\n    ','}')
    html_data4 = html_data4.text.replace('\n    {\n        ','{')
    html_data4 = html_data4.replace('\n    }\n    ','}')
    pre_date = pd.to_datetime(html_data3[17:])
    pre_price = html_data4[1:]

    date = []
    crude = []
    date.append(pre_date)
    crude.append(round(float(pre_price),2))
    df2["Date"] = date
    df2["Crude Closing"] = crude

    oil_df = pd.read_csv("Resources/data/oil-prices.csv")
    oil_df['Date'] = pd.to_datetime(oil_df['Date'])
    oil_df = pd.concat([oil_df, df2], ignore_index = True)
    oil_df = oil_df.drop(columns=['Unnamed: 0'])
    oil_df = oil_df.drop_duplicates()
    oil_df.to_csv('Resources/data/oil-prices.csv')

    # Merge DataFrames

    oil_gas_df = pd.merge(gas, oil_df, on=["Date", "Date"])
    oil_gas_df.sort_values(by=['Date'], inplace=True)

    # Convert gas price data dataframe to dictionary.
    oil_gas_df.index = oil_gas_df.index.map(str)
    oil_gas_dict = oil_gas_df.to_dict()

    current_gas_dict = df.iloc[0].to_dict()
    latest_crude_price = oil_gas_df.iloc[-1][-1]

    return latest_crude_price, current_gas_dict, oil_gas_dict
