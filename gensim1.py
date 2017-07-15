import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim

from gensim import corpora, models, similarities
corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
[(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
[(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
[(0, 1.0), (4, 2.0), (7, 1.0)],
[(3, 1.0), (5, 1.0), (6, 1.0)],
[(9, 1.0)],
[(9, 1.0), (10, 1.0)],
[(9, 1.0), (10, 1.0), (11, 1.0)],
[(8, 1.0), (10, 1.0), (11, 1.0)]]

tfidf = models.TfidfModel(corpus)
vec = [(0, 1), (4, 1)]
print(tfidf[vec])
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)
sims = index[tfidf[vec]]
print(list(enumerate(sims)))

# import modules and set up logging
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# load up unzipped corpus from http://mattmahoney.net/dc/text8.zip
sentences = word2vec.Text8Corpus('./text8')
# train the skip-gram model; default window=5
model = word2vec.Word2Vec(sentences, size=200)
# ... and some hours later... just as advertised...
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
[('queen', 0.5359965)]
 
# pickle the entire model to disk, so we can load&resume training later
model.save('./text8.model')
# store the learned weights, in a format the original C tool understands
model.save_word2vec_format('/tmp/text8.model.bin', binary=True)
# or, import word weights created by the (faster) C word2vec
# this way, you can switch between the C/Python toolkits easily
model = word2vec.Word2Vec.load_word2vec_format('/tmp/vectors.bin', binary=True)
 
# "boy" is to "father" as "girl" is to ...?
model.most_similar(['girl', 'father'], ['boy'], topn=3)
[('mother', 0.61849487), ('wife', 0.57972813), ('daughter', 0.56296098)]
more_examples = ["he his she", "big bigger bad", "going went being"]
for example in more_examples:
    a, b, x = example.split()
    predicted = model.most_similar([x, b], [a])[0][0]
    print "'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted)
# which word doesn't go with the others?
model.doesnt_match("breakfast cereal dinner lunch".split())
