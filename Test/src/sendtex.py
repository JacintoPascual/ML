import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
# import numpy as np


digits = datasets.load_digits()
print(type(digits))  # Bunch
print(type(digits.data))  # numpy.ndarray
print(type(digits.target))  # numpy.ndarray

# print(digits.data)
# print(digits.target)
# print(digits.images[0])

clf = svm.SVC(gamma=0.001, C=100)
print(len(digits.data))

# We load the data except the last one. 10 in this last example
# X Capitalized because it represent a Matrix (#rows x #columns)
# y Represent a vector (#rows)
X, y = digits.data[:-10], digits.target[:-10]
print(X.shape)
print(y.shape)

# print(X.shape)
# X = y.reshape([1,-1])
clf.fit(X, y)
# prediction = clf.predict(digits.data[4])
print(digits.data[4])
# print('Prediction', clf.predict(digits.data[4]))
# X = X.reshape([1, -1])
# print(X)
print(X[0])
print('Prediction', clf.predict(digits.data[4]))
# print(clf.predict([1, 1]))
# clf.predict(np.array([1,1, 1, 1]))

# plt.imshow(digits.images[-4], cmap=plt.cm.gray_r, interpolation="nearest")
# plt.show()

# ##################################################
#
# This part is really from the Data School videos
# It will be continued in the regression2.py
#
# ##################################################

# Tran/test split procedure for model validation
# Training accuracy when you train and test the model on the same data
# Not the best procedure since it would not necessarily predict well on Out Of Sample data
y_pred = clf.predict(X)
print(metrics.accuracy_score(y, y_pred))
# It is better with a train/split evaluation procedure
# We reserve some of our data for testing, the other ones for training
# Here we reserve the last 10 rows for testing set
# X, y = digits.data[:-10], digits.target[:-10]
# Using sklearn model_selection.train_test_split. Test size means that 20% are for testing
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
# To always split on the same way/data random_test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=4)
y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

# For which K-value is best in the KNeighborsClassifier build a for loop
k_range = range(1, 26)
scores = []
#  for k in k_range:
#    knn = KNeighborsClassifier
#    knn.fit(X_train, y_train)
#    y_pred = knn.predict(X_test)
#    scores.append(metrics.accuracy_score(y_test, y_pred))
# And use matplotlib to visualize
# %matplotlib inline  ???
# plt.plot(k_range, scores)
# plt.xlabel("Value of k for KNN"
# plt.ylabel("Testing accuracy")
# After decided the model train your model with all the data X otherwise we lose valuable training data
# cls.fit(X, y)
# and we predict with an out-of-range observation that we can construct manually
#  knn.predict([.., ..., ..., ...]


# ### Continue in regression2.py





