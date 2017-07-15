# -*- coding: utf-8 -*-
import nltk
from collections import Counter
from nltk.corpus import stopwords

from urllib import urlopen
surl = 'http://www.gutenberg.org/files/2554/2554.txt'
raw = urlopen(surl).read()
word_list = nltk.word_tokenize(raw)
word_list = [word.lower() for word in word_list]
print len(word_list)
stoplist = list(stopwords.words('english'))
print len(stoplist)
stoplist.append("the")
stoplist.append('.');stoplist.append("''");stoplist.append("``");
stoplist.append('...');stoplist.append('?');stoplist.append('!');stoplist.append(';')
stoplist.append('--');stoplist.append('\'s')
stoplist.append(',')
print len(stoplist)
word_list = [word for word in word_list if word not in stoplist]
print len(word_list)
count_words = Counter(word_list)
print count_words.most_common(50)
