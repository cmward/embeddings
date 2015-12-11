from gensim.models import Word2Vec
from corpus import SubdirSentences
import climate

""" Train a model using gensim.
"""

climate.enable_default_logging()

sentences = SubdirSentences('../data/text', n_files=5500)
model = Word2Vec(sentences, size=100, window=10, min_count=20, sample=1e-5,
                 workers=4, negative=5, hs=0, sg=1)
model.save('../models/gs100')
