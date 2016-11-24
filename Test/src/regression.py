
# sentdex youtube video
import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm, model_selection
# cross_validation deprecated and replaced by model_selection
from sklearn.linear_model import LinearRegression


df = quandl.get('WIKI/GOOGL')

# print(df.head())
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
# print(df.head())

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

# test_size means that we keep 20% of the data for testing set
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
# To run on 10 threads
# clf = LinearRegression(n_jobs=10)
# To run on that many threads as possible by your processor
# clf = LinearRegression(n_jobs=-1)
# To use another algorithm
# clf = svm.SVR()  # This gave worse accuracy 0.79###
# clf = svm.SVR(kernel='poly')  # And even worse accuracy 0.5134###
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)  # 0.95###
