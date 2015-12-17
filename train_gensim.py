from gensim.models import Word2Vec
from corpus import SubdirSentences
import climate

""" Train a model using gensim.
"""

climate.enable_default_logging()

sentences = SubdirSentences('../data/text', n_files=5500)
for dim in [50, 100]:
    model = Word2Vec(sentences, size=dim, window=10, min_count=20, sample=1e-5,
                     workers=4, negative=5, hs=0, sg=1)
    model.save('../models/gs{}'.format(dim))
