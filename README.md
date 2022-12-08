## Gasoline Price Predictor Machine Learning Models: Linear Regression and Natural Language Processsing (Sentiment Analysis)
### Linear Regression (see gas_oil_ML.ipynb)
#### Description of Preliminary Data Preprocessing
Starting with crude oil data (1983 to present) and gas and diesel price data (1995 to 2021), the data are cleaned to fit into a linear regression machine learning model.
The data are loaded into two dataframes (gas_df and crude_df). The cleaned gas price dataframe includes all formulations of retail gasoline and diesel prices in a MM/DD/YYYY format with samples from each month starting in January 1995 to January 2021. The cleaned crude oil dataframe is in the same format as the cleaned gas price dataframe: MM/DD/YYYY format with monthly samples from January 1995 to January 2021.
#### Description of Preliminary Feature Engineering and Preliminary Feature Selection
To fit a linear regression model with crude oil and gas price data to establish and explore relationships between the data.
#### Explanation of Model Choice, including Limitations and Benefits 
This linear regression model is the best model because historically, crude oil prices and gas prices have a linear relationship, so there is no need to complicate the relationship with other, more complex models.
#### Train, Test, Split Description 
The data are split into 80% training and 20% testing.

### Natural Language Processsing (Sentiment Analysis) (see twitter_NLP.ipynb)
#### Description of Preliminary Data Preprocessing
Using tweets from November 2022 to present, the data are cleaned and subjectivity and polarity score columns are included.
#### Description of Preliminary Feature Engineering and Preliminary Feature Selection
To use natural language processing to characterize the sentiment of current tweets on gas prices (positive, neutral, or negative sentiment).
#### Explanation of Model Choice, including Limitations and Benefits 
This sentiment analysis NLP model is useful to understand consumers' feedback and their needs. A, perhaps, better NLP model would be a BERT model. A BERT model has two tasks: language modeling and next sentence prediction, which is better for the context of tweets, rather than a simple positive, neutral, or negative sentiment.
#### Train, Test, Split Description 
The data are split into 80% training and 20% testing.
