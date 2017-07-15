'''
concept: similar, trigrams
checks for similar words 
'''
str = "my name is sreedhar. my name denver sreedhar. city name hundred sreedhar"
print str
str_words = nltk.word_tokenize(str)
text = nltk.Text(word.lower() for word in str_words)
print "similar word for 'is'  --- ",text.similar('is')
str_tg_list = list(trigrams(str_words))
print "trigrams list is",str_tg_list
t = [item for item in str_tg_list if 'is' in item]
t = [item for item in str_tg_list if 'is' in item and 'name' in item]
print t