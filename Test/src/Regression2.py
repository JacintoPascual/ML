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
print("Printing head:\n", data.head())
# print(data.tail())
print(data.shape)  # (200, 4)
print("Printing sales over 20\n", data[data.Sales > 20])
print("Printing only TV when sales over 20\n", data.loc[data.Sales > 20, 'TV'])
# We can also "isin" with string type
# movies[movies.genre.isin(['Crime', 'Drama', 'Action'])].head(10)
# various methods are available to iterate through a DataFrame
for index, row in data.iterrows():
    print(index, data.TV, data.Radio)
# From another example
# only include numeric columns in the DataFrame
# drinks.select_dtypes(include=[np.number]).dtypes
# drinks.describe(include='all')
# pass a list of data types to only describe certain types
# drinks.describe(include=['object', 'float64'])
# you can interact with any DataFrame using its index and columns
# drinks.describe().loc['25%', 'beer_servings']
# 'index' is an alias for axis 0 and 'columns' is an alias for axis 1
# drinks.mean(axis='index')
# drinks.mean(axis='columns').head()
# many pandas string methods support regular expressions (regex)
# orders.choice_description.str.replace('[\[\]]', '').head()
#
# calculate the mean beer servings just for countries in Africa
# drinks[drinks.continent=='Africa'].beer_servings.mean()
# calculate the mean beer servings for each continent
# drinks.groupby('continent').beer_servings.mean()
# other aggregation functions (such as 'max') can also be used with groupby
# drinks.groupby('continent').beer_servings.max()
# multiple aggregation functions can be applied simultaneously
# drinks.groupby('continent').beer_servings.agg(['count', 'mean', 'min', 'max'])
# display percentages instead of raw counts
# movies.genre.value_counts(normalize=True)
# count the number of missing values in each Series
# movies.isnull().sum()
# use the 'isnull' Series method to filter the DataFrame rows
# movies[movies.genre.isnull()].head()
# if 'any' values are missing in a row, then drop that row
# ufo.dropna(how='any').shape
# if 'all' values are missing in a row, then drop that row (none are dropped in this case)
# ufo.dropna(how='all').shape
# if 'any' values are missing in a row (considering only 'City' and 'Shape Reported'), then drop that row
# ufo.dropna(subset=['City', 'Shape Reported'], how='any').shape
# explicitly include missing values
# ufo['Shape Reported'].value_counts(dropna=False).head()
# fill in missing values with a specified value
# ufo['Shape Reported'].fillna(value='VARIOUS', inplace=True)
# create a small DataFrame from a dictionary
# df = pd.DataFrame({'ID':[100, 101, 102, 103], 'quality':['good', 'very good', 'good', 'excellent']})
#
# create a DataFrame of passenger IDs and testing set predictions
# ensure that PassengerID is the first column by setting it as the index
# write the DataFrame to a CSV file that can be submitted to Kaggle
# pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': new_pred_class}).set_index('PassengerId').to_csv('sub.csv')
# save a DataFrame to disk ("pickle it")
# train.to_pickle('train.pkl')
# read a pickled object from disk ("unpickle it")
# pd.read_pickle('train.pkl').head()
#
# sample 75% of the DataFrame's rows without replacement
# train = ufo.sample(frac=0.75, random_state=99)
# store the remaining 25% of the rows in another DataFrame
# test = ufo.loc[~ufo.index.isin(train.index), :]
#
# use the 'drop_first' parameter (new in pandas 0.18) to drop the first dummy variable for each feature
# pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True).head()
#
# read a dataset of movie reviewers into a DataFrame
user_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_table('http://bit.ly/movieusers', sep='|', header=None, names=user_cols, index_col='user_id')
# detect duplicate zip codes: True if an item is identical to a previous item
users.zip_code.duplicated().tail()
# count the duplicate items (True becomes 1, False becomes 0)
users.zip_code.duplicated().sum()
# detect duplicate DataFrame rows: True if an entire row is identical to a previous row
users.duplicated().tail()
# examine the duplicate rows (ignoring the first occurrence)
users.loc[users.duplicated(keep='first'), :]
# drop the duplicate rows (inplace=False by default)
users.drop_duplicates(keep='first').shape
# only consider a subset of columns when identifying duplicates
users.duplicated(subset=['age', 'zip_code']).sum()
#
# view the option descriptions (including the default and current values)
pd.describe_option()



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
