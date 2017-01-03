import pandas as pd
import logging
from datetime import date
from string import Template

t = Template('${village}folk send $$10 to $cause.')
s = t.substitute(village='Nottingham', cause='the ditch fund')
print(s)  # 'Nottinghamfolk send $10 to the ditch fund.'

# The substitute() method raises a KeyError when a placeholder is not supplied
# in a dictionary or a keyword argument
# For mail-merge style applications, the safe_substitute() method may be more appropriate
# it will leave placeholders unchanged if data is missing:
t = Template('Return the $item to $owner.')
d = dict(item='unladen swallow')
s = t.safe_substitute(d)
print(s)  # 'Return the unladen swallow to $owner.'


# Debug and info are by default suppressed
logging.debug('Debugging information')
logging.info('Informational message')
# logging.warning('Warning:config file %s not found', 'server.conf')
# logging.error('Error occurred')
# logging.critical('Critical error -- shutting down')

now = date.today()
print("Formated date now: ", now.strftime("%m-%d-%y. %d %b %Y is a %A on the %d day of %B."))
# '12-02-03. 02 Dec 2003 is a Tuesday on the 02 day of December.'
print("Unformated date now: ", now)

# *********************************************************************
# lambda
a = [(lambda x: x*x)(x) for x in range(10)]
print("Printing a version 1: ", a)
a = [x*x for x in range(10)]
print("Printing a version 2: ", a)
a = map(lambda x: x*x, range(10))
print("Printing map: ", a)
a = list(a)
print("Printing list from map: ", a)
print("Printing list item from map: ", a[2])


# List comprehension
x = [2, 3, 4, 5, 6]
y = [v * 5 for v in x]
print("Printing y list: ", y)
# Same result. Map takes a function and an iterable and then return another iterable.
y = map(lambda v: v*5, x)
print("Printing y map: ", y)
print("Printing y list from map: ", list(y))


# Multiply odd values and add to list
y = [v * 5 for v in x if v % 2]
# filter() takes in a function (lambda) and iterable(x) and return an iterable containing "filtered" values
y = map(lambda v: v*5, filter(lambda u: u % 2, x))
y = list(y)
print("Printing y list of odds from map: ", y)


x = [2, 3, 4]
y = [4, 5]
z = []
for v in x:
    for w in y:
        z += [v + w]
assert z == [2+4, 2+5, 3+4, 3+5, 4+4, 4+5]
assert z == [6, 7, 7, 8, 8, 9]
# With list comprehension
z = [v + w for v in x for w in y]
assert z == [6, 7, 7, 8, 8, 9]

# With "pseudo map and lambda"
# map(lambda for v: in map(lambda for w: v + w, in y), in x)
# With map and lambda
t = map(lambda v: map(lambda g: v+g, y), x)
# *********** NOT AS EXPECTED ****************
# *********** Must be reviewed later ****************
print("Printing list from map t", list(t))
print("Printing t==(v1)", t == [[6, 7], [7, 8], [8, 9]])
print("Printing t==(v2)", list(t) == [[6, 7], [7, 8], [8, 9]])
print("Printing z", z)
# z = sum(t, [])
print("Printing t==(v1)", z == [2+4, 2+5, 3+4, 3+5, 4+4, 4+5])
print("Printing t==(v2)", z == [6, 7, 7, 8, 8, 9])


# We can also "isin" with string type
# movies[movies.genre.isin(['Crime', 'Drama', 'Action'])].head(10)

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
# print("Duplicated zip codes\n", users.zip_code.duplicated().tail())
print("Users with duplicated zip codes\n", users[users.zip_code.duplicated()].sort_values)
# print("Users with duplicated zip codes\n", users[users.zip_code.duplicated()].sort_values(by=users.occupation))
# count the duplicate items (True becomes 1, False becomes 0)
users.zip_code.duplicated().sum()
# detect duplicate DataFrame rows: True if an entire row is identical to a previous ro
users.duplicated().tail()
# examine the duplicate rows (ignoring the first occurrence)
users.loc[users.duplicated(keep='first'), :]
# drop the duplicate rows (inplace=False by default)
users.drop_duplicates(keep='first').shape
# only consider a subset of columns when identifying duplicates
users.duplicated(subset=['age', 'zip_code']).sum()
#
# view the option descriptions (including the default and current values)
# pd.describe_option()


