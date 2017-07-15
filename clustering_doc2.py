from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time
import numpy as np

#Convert a collection of text documents to a matrix of token counts
#
#
#
categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space']
#categories = ['alt.atheism']
print(categories)
twenty_train = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
print("%d documents" % len(twenty_train.data))
print("%d categories" % len(twenty_train.target_names))
print()

labels = twenty_train.target
true_k = np.unique(labels).shape[0]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
# returns a term-document matrix. Learns Vocabulary dictionary
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)
#count_vect.vocabulary_.get(u'algorithm')
zip1 = zip(count_vect.get_feature_names(),np.asarray(X_train_counts.sum(axis=0)).ravel())
dict1 = dict(zip1)
#print( type(dict1))
#print(repr(dict1).encode("utf-8")) # prints each word & its frequency

# fit_transform 2 steps into one
#Transform a count matrix to a normalized tf or tf-idf representation 
tfidf_transformer = TfidfTransformer(use_idf = True, smooth_idf = True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(repr(tfidf_transformer))
print("shape of tfidf matrix",X_train_tfidf.shape)
from sklearn.naive_bayes import MultinomialNB
# alpha, fit_prior, class_prior
clf = MultinomialNB(alpha = 1, fit_prior = True).fit(X_train_tfidf, twenty_train.target)
print(clf)
#need to use the same feature extracting chain before
docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
X1_new_counts = count_vect.transform(docs_test)
X_new_tfidf = tfidf_transformer.transform(X1_new_counts)
predicted2 = clf.predict(X_new_tfidf)
print(np.mean(predicted2 == twenty_test.target))
# this is only for multi-label classification
score1 = clf.score(X1_new_counts, twenty_test.target)
print("clf.score >> >> ",score1)
