
list1 = ['wd1','sd1','kd1']
list2 = ['wd2','sd2','kd2']
[(item1,item2) for item1 in list1 for item2 in list2 ]

[(len(word),word) for item in mylist]

cfd = nltk.ConditionalFreqDist( (genre, word)
for genre in brown.categories()
for word in brown.words(categories=genre))

genre_word = [ (genre, word)
for genre in brown.categories()
for word in brown.words(categories=genre) ]