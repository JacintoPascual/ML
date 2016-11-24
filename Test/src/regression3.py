# Features are always numbers. In case of text they mus be  "converted" to numbers
from sklearn.feature_extraction.text import CountVectorizer

# Example text for model training (SMS messages)
simple_train = ['call you tonight', 'call me a cab', 'pleae call me... please']
vect = CountVectorizer()
# Learn the "vocabulary" of the training data
vect.fit(simple_train)

# Examine the vector
vect.get_feature_names()

# Transform training data into a 'document-term
simple_train_dtm = vect.transform(simple_train)
simple_train

# convert sparse matrix to a dense matrix
simple_train
# Just for git testing

