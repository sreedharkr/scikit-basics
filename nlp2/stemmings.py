from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer


#stemmer = PorterStemmer()
stemmer = SnowballStemmer("english")
plurals = ['caresses', 'flies', 'dies', 'mules', 'denied',
   'died', 'agreed', 'owned', 'humbled', 'sized',
           'meeting', 'stating', 'siezing', 'itemization',
            'sensational', 'traditional', 'reference', 'colonizer',
          'plotted']
print(plurals)
singles = [stemmer.stem(plural) for plural in plurals]
print(singles)