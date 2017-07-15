#execfile('helloworld.py')

str = ' Human is influenced by past all the time. Student is influenced by teacher.\n  Artist is influenced by nature.\n  I am influenced by philosophy'

tokens = nltk.word_tokenize(str)

#Create your bigrams
bgs = nltk.bigrams(tokens)

#compute frequency distribution for all the bigrams in the text
fdist = nltk.FreqDist(bgs)
for k,v in fdist.items():
    print k,v