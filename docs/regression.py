# Regression - Algorithm for Stock Prices

# Note: Features are the attributes that make up the label, and the label is
# hopefully a strong prediction of a future value. Usually want between -1 and 1.

import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

# manipulating dataframe to only show this data
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

# Calculate percent volatility
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'] / df['Adj. Close'] * 100.0)
#calculate daily percent change
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open'] / df['Adj. Open'] * 100.0)

# define new dataframe - note: volume is the amt of trades made during the day
df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]

# Now we will define a label - we want the Adj. Close of the future
forecast_col = 'Adj. Close'
# filling in case data is missed, and this will treat NAN's as outliers
# can't work with NAN data in machine learning
df.fillna(-99999, inplace=True)

# trying to predict out 10% of the dataframe
# using data from 10 days ago to predict 10 days out
forecast_out = int(math.ceil(0.01*len(df)))

# shifting columsn in a negative manner so that each label column of each row
# will be the adjusted row price 10 days into the future
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)


# Now going to define features and pass them to a classifier
# X = features; y = labels

X = np.array(df.drop(['label'], 1)) #returns new dataframe
y = np.array(df['label'])

X = preprocessing.scale(X) #scale from -1 to 1, would be skipped in high frequency trading
df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# Define classifier
clf = LinearRegression(n_jobs=-1) #97% accuracy

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
