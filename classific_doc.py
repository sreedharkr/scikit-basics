import nltk 
import random
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
documents = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

list_stop = stopwords.words('english')
list2 = [',','-','.','\'','""','(',')','?','"',':',';']
list_stop.extend(list2)
all_words_stop = [a for a in movie_reviews.words() if a not in list_stop] 
#all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
all_words = nltk.FreqDist(w.lower() for w in all_words_stop)
print("len(all_words)",len(all_words))
word_features = list(all_words.keys())[:2000]
print(word_features[:5])


word_features2 = all_words.most_common(2000)
word_features3 = [a[0]for a in word_features2]
print("most common::",word_features[:5])
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
        return features

featuresets = [(document_features(d), c) for (d,c) in documents]
print("len() featuresets::",len(featuresets))
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
