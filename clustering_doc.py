from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
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
categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space']
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()

labels = dataset.target
true_k = np.unique(labels).shape[0]
#max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
#max_df = 25 means "ignore terms that appear in more than 25 documents".
#min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
#min_df = 5 means "ignore terms that appear in less than 5 documents".
vectorizer = TfidfVectorizer(max_df=0.6, max_features=3387,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(dataset.data)

#print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

svd = TruncatedSVD(2500,random_state = 222)
#print(repr(svd))
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd,normalizer)
X = lsa.fit_transform(X)
explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))
#k-means++ , random
km = KMeans(n_clusters=4, init='random', max_iter=300, n_init=25,
                verbose=False, random_state = 222)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))
#print(metrics.accuracy_score(labels, km.labels_))
arr1 = np.c_[labels, km.labels_]
