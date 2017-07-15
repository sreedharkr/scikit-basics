def gender_features(word):
    return {'last_letter': word[-1]}

from nltk.corpus import names
import random
names2 = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(names2)
print(len(names2))
print(names2[0:10])
featuresets = [(gender_features(n), g) for (n, g) in names2]
print(len(featuresets))
print(featuresets[0:10])
train_set, test_set = featuresets[500:], featuresets[:500]
from nltk import NaiveBayesClassifier
nb_classifier = NaiveBayesClassifier.train(train_set)
print(nb_classifier.classify(gender_features('Gary')))
print(nb_classifier.classify(gender_features('Grace')))

from nltk import classify
print(classify.accuracy(nb_classifier, test_set))



