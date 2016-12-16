import pandas as pd
import seaborn as sns  # On top of matplotlib
# import matplotlib
# import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# #####################################################
# Regression problems using pandas for reading csv file
# #####################################################

# data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv")
# pandas figure out how to make a matrix out of the csv
# First column (Unnamed) is probably the Id for the observation
# We set the column 0 as the index
# There are many parameters that we can use in the read_csv function, ex.: index_col
data = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv", index_col=0)

# Display the first and last 5 rows
print(data.head())
# print(data.tail())
print(data.shape)  # (200, 4)


# The data is the advertising dollars spent on a single ads for different media in a single market
# The index are the market
# The target (sales) is sales of a single product in a given market (in thousands of item)
# Because the response variable is continuous, this is a regression problem

# Allow plots to appear within iPython notebook (Seems not to work within PyCharm)
# %matplotlib inline

# Visualize the relationship between the features and the response using scatterplots
# Not starting plot from IDE, in the example he used iPython notebook
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')
# Seems to be a good candidate for Linear regression based on the line drawn in the plots
# when we used the kind='reg' parameter

# Panda is build on top of NumPy, thus X can be a pandas DataFrame and y a pandas Series
features_cols = ['TV', 'Radio', 'Newspaper']
# Tells panda to take a subset of the columns
X = data[features_cols]
print("Printing head of a subset of the pandas DataFrame")
print(X.head())
# Equivalent to do this
# The outer brake tells panda that you want to take a subset of dataFrame columns
# And inner brake defines a Python list
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
# Equivalent command that works if there are no spaces in the column name
# y = data.Sales

# test_size means that we keep 20% of the data for testing set
# Default split is 75%
# random_state makes that the train and test selections are always the same
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Instantiate the model
linReg = LinearRegression()
# Train the model
linReg.fit(X_train, y_train)
# Formula for linear regression
# y = B0 + B1X1 + B2X2 + B3X3 + ... BnXn
# y = B0 + B1 TV + B2 Radio + B3 Newspaper, B0 is the value of y when all the coefficients (B#) are 0
# and it is called the interceptor.
# These values are "learned" during the model fitting step using the "least square" criterion.
# Then, the fitted model can be used to make predictions!
print(linReg.intercept_)
print(linReg.coef_)
# The trailing '_' is a convention that means that the data is estimated

# zip pairs to tuple of the same size
# Pair the feature names with the coefficients
# Otherwise the coefficients can be hard to read. Which belongs to which
zip(features_cols, linReg.coef_)
print(list(zip(features_cols, linReg.coef_)))
# In the linear formula it would be something like this
# y = 2.88 + 0.0466 * TV + 0.179 * Radio + 0.00345 * Newspaper
# For a given amount of Radio and Newspaper ad spending,
# a "unit" increase in TV ad spending is associated with a 0.0466 "unit" increase in Sales
# OR an increase of additional $1,000 spent on TV ads is associated with an increase on Sales of 46,6 items
# We talk about "association", not #causation" since they can be other factors that we do not know
# If an increase in TV ad spending was associated with a decrease in sales, B1 (in the formula) would be negative

# Make prediction on the testing test
y_pred = linReg.predict(X_test)

# Model evaluation metric for regression
# Evaluation metrics for classification problems, such as accuracy, are not useful for regression problems.
# Instead, we need evaluation metrics designed for comparing continuous values
# Let's create some example numeric predictions, and calculate three common evaluation metrics for regression problems
# Define true and predicated response values
true = [100, 50, 30, 20]
pred = [90, 50, 50, 30]
# The error here are 10 (100-90), 0 (50-50), 20 (30-50) and 10 (20-30)

# Calculate MAE (Mean Absolute Error) by hand
print((10 + 0 + 20 + 10)/4.)
# Calculate MAE (Mean Absolute Error) using scikit-learn
print(metrics.mean_absolute_error(true, pred))

# Calculate the Mean Square Error (MSE) by hand
print((10**2 + 0**2 + 20**2 + 10**2)/4.)
# Calculate MSE using scikit-learn
print(metrics.mean_squared_error(true, pred))

# Calculate the Root Mean Square Error (MSE) is the mean of the squared errors
# By hand
print(np.sqrt((10**2 + 0**2 + 20**2 + 10**2)/4.))
# Calculate RMSE using scikit-learn
print(np.sqrt(metrics.mean_squared_error(true, pred)))

# MAE is the easiest to understand, because it's the average error
# MSE is more popular than MAE, because MSE "punishes" larger errors, since we square
# RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.

# Computing the RMSE for our Sales prediction
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Feature selection
# Does Newspaper "belongs" to out model
features_cols = ['TV', 'Radio']
X = data[features_cols]
y = data.Sales

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=1)
linReg.fit(X_train, y_train)
y_pred = linReg.predict(X_test)
# Computing the RMSE for our Sales prediction
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# The RMSE decreased when we removed Newspaper from the model.
# Error is something we want to minimize, so a lower number for RMSE is better.
# Thus, it is unlikely that this feature is useful for predicting Sales, and should be removed from the model.
# We could repeat this process with other combinations of features...
