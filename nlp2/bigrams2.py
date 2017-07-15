# -*- coding: utf-8 -*-
# most frequently word occurrences before and after a given word in this text.

text = 'Human is influenced by past all the time. Student is influenced by teacher.\n  Artist is influenced seriously by nature.\n  I am influenced by philosophy'.split()

from nltk import bigrams
bgs = bigrams(text)
lake_bgs = filter(lambda item: item[0] =='influenced', bgs)
print(lake_bgs)
print len(lake_bgs) 
from collections import Counter
c = Counter(map(lambda item: item[1], lake_bgs))
print c
print Counter(map(lambda item: item[1], lake_bgs))
print c.most_common()