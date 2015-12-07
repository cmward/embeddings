from __future__ import division
from collections import defaultdict
from itertools import islice
from scipy.optimize import check_grad
from scipy.special import expit
from cPickle import dump, load, HIGHEST_PROTOCOL as HIGHEST_PICKLE_PROTOCOL

import numpy as np
import random
import copy
import math
import sys
import time

from corpus import Sentences

def sigmoid(z):
    # Clip values to be within [-10,10] to avoid overflow.
    z = np.clip(z, -10, 10)
    return expit(z)

def cos(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def reverse_enum(sequence, start=0):
    n = start
    for elem in sequence:
        yield elem, n
        n += 1

def window(seq, n=5):
    # https://docs.python.org/release/2.3.5/lib/itertools-example.html
    """Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = list(islice(it, n))
    if len(result) == n:
        yield result    
    for elem in it:
        result = result[1:] + [elem]
        yield result

class Vocab(object):
    def __init__(self, corpus):
        self.corpus = corpus # generator
        self.word_map = {}
        self.index_map = {}
        self.count = defaultdict(int)
        self.total = 0
    
    def build_vocab(self, min_count=10):
        """
        :type min_count: int
        :param min_count: discard words that occur fewer than `min_count` times
        """
        print "Building vocab..." 
        self.total = 0
        index = 0
        for sentence in self.corpus:
            for word in sentence:
                if word not in self.word_map:
                    self.word_map[word] = index
                    index += 1
                self.count[word] += 1
                self.total += 1
        self.index_map = dict((v, k) for k, v in self.word_map.iteritems())
        print "Vocab contains {} words.".format(len(self.word_map))
        to_delete = [w for w in self.count if self.count[w] < min_count]
        if len(to_delete) > 0:
            for w in to_delete:
                del self.word_map[w]
                del self.count[w]
            tmp_map = copy.deepcopy(self.word_map)
            self.word_map = dict(reverse_enum(tmp_map.keys()))
            self.index_map = dict((v, k) for k, v in self.word_map.iteritems())
            print "min_count removed {} words, {} words remaining in vocab.".format(
                    len(to_delete), len(self.word_map))

    def __len__(self): return len(self.word_map)
    def __iter__(self): return self.word_map.iterkeys()

class UnigramTable(object):
    """
    List of word indices from the vocab, used to draw samples in negative
    sampling. Unigram distribution raised to 3/4th power.
    Copied more or less line for line from the original word2vec code.
    """
    def __init__(self, vocab):
        print "Populating unigram table..."
        table_size = 1e8 
        table = np.zeros((table_size))
        vocab_size = len(vocab)
        power = 0.75
        train_words_pow = sum([math.pow(vocab.count[w], power) for w in vocab])
        d1 = 0 
        i = 0
        for w,j in vocab.word_map.iteritems():
            d1 += math.pow(vocab.count[w], power)/train_words_pow
            while i < table_size and i / table_size < d1:
                table[i] = j
                i += 1
        self.table = table

    def sample(self, neg):
        indices = np.random.randint(low=0, high=len(self.table), size=neg)
        return [self.table[i] for i in indices]

class Embeddings(object):
    def __init__(self, vocab, dim=100, context_size=4, neg=5, syn0=None, 
                 syn1=None):
        """
        :type vocab: Vocab
        :param vocab: words to learn embeddings for

        :type dim: int
        :param dim: size of hidden layer (dimensionality of embeddings)

        :type context_size: int
        :param context_size: number of words to consider in the context
        around the input word

        :type neg: int
        :param neg: how many samples to draw for negative sampling

        :type syn0: np.ndarray
        :param syn0: input->hidden weight matrix, shape (|V|, dim)

        :type syn1: np.ndarray
        :param syn1: hidden->output weight matrix, shape(dim, |V|),

        """
        if isinstance(vocab, Vocab):
            self.vocab = vocab
        else:
            with open(vocab) as v_file:
                self.vocab = load(v_file)

        self.word_map = self.vocab.word_map
        self.index_map = self.vocab.index_map
        self.dim = dim
        self.v_size = len(vocab)
        self.context_size = context_size
        self.neg = neg

        # Initialize weights randomly
        if syn0 is None:
            syn0 = np.random.uniform(
                low=-1./(2*self.dim),
                high=1./(2*self.dim),
                size=(self.v_size,self.dim))
            self.syn0 = syn0
        else: 
            self.syn0 = np.load(syn0)

        if syn1 is None:
            syn1 = np.random.uniform(
                low=-1./(2*self.dim),
                high=1./(2*self.dim),
                size=(self.dim,self.v_size))
            self.syn1 = syn1
        else:
            self.syn1 = np.load(syn1)

    def save(self, syn1=False):
        """
        Save the model to a .npy file.
        syn1=True will also save syn1 to a separate file, so that
        training can be resumed on the model.
        """
        if syn1 is True:
            np.save("syn1.npy", self.syn1)
        np.save("syn0.npy", self.syn0)
        with open("vocab","w+") as v_file:
            dump(self.vocab, v_file, HIGHEST_PICKLE_PROTOCOL) 

    def v(self, word):
        """Get the vector for a word."""
        return self.syn0[self.word_map[word]]
    
    def train_pair(self, input_word, target_word, samples, theta,
                   learning_rate, table):
        """
        Given an input word predict a single context word.
        Calculate the error using negative sampling.

        :type input_word: int
        :param input_vector: word index used as input to network

        :type target_word: int
        :param target_word: gold standard output word index

        :type theta: np.ndarray
        :param theta: np.dot(self.syn1.T[samples], h), input to the output
        layer units for target and sample words

        :type learning_rate: int
        :param learning_rate: learning rate
        """
        # Hidden layer h is the row in the input->hidden weight 
        # matrix syn0 corresponding to the input word.
        # theta is output layer input.
        # z is the output layer output.
        # g_z is the gradient wrt z.
        # g_w is the gradient wrt syn1.
        # g_h is the gradient wrt theta (hidden layer output).
        labels = [0 if sample != target_word else 1 for sample in samples]
        z = sigmoid(theta) # shape (context_size+1,)
        g_z = z - labels # shape (context_size+1,)
        g_w = np.outer(g_z, input_word) # shape (context_size+1, dim)
        self.syn1.T[samples] -= learning_rate * g_w # update output vectors
        g_h = np.dot(g_z, self.syn1.T[samples]) # shape (dim,)
        return g_h

    def train(self, corpus, learning_rate, table=None):
        """
        :type corpus: Sentences
        :param corpus: iterable of sentences

        :type learning_rate: int
        :param learning_rate: multiply gradients by learning_rate before update

        :type table: UnigramTable
        :param table: unigram distribution to randomly sample words from for
        negative sampling
        """
        if table is None:
            table = UnigramTable(self.vocab)
        wc = 0
        for sentence in corpus:
            for word in sentence:
                wc += 1
                if wc % 1000 == 0:
                    log = "{} PROGRESS: {:.0f} of {:.0f} words, {:.2%}".format(
                        time.strftime("%H:%M:%S"),
                        wc, corpus.word_count,
                        (wc / corpus.word_count))
                    print log
            # sliding windows of size context_size + 1
            windows = window(sentence, self.context_size+1)
            for wind in windows:
                input_word = wind[self.context_size // 2]
                if input_word not in self.word_map:
                    continue
                else:
                    input_index = self.word_map[input_word]
                    for word in wind:
                        if word not in self.word_map:
                            wind[wind.index(word)] = None
                    contexts = [self.word_map[c] for c in wind 
                                if (c is not None and c != input_index)]
                    if len(contexts) <= 1:
                        continue
                    g_h = np.zeros((self.dim))
                    for context in contexts:
                        samples = table.sample(self.neg)
                        samples.append(context)
                        h = self.syn0[input_index]
                        theta = np.dot(self.syn1.T[samples], h)
                        g_h += self.train_pair(
                            h, context, samples, theta, learning_rate, 
                            table)
                    # update embedding for input
                    self.syn0[input_index] -= learning_rate * g_h

    def most_similar(self, word, n=5):
        """
        Find the n most similar words to `word`.

        Computes the cosine similarity between `word` and every other
        word in the vocabulary. 
        """
        if isinstance(word, np.ndarray):
            v1 = word
        else:
            v1 = self.v(word)
        sims = np.apply_along_axis(lambda v2: cos(v1, v2), 1, self.syn0)
        highest_sims = sims.argsort()[-n-1:][::-1][1:]
        return [(self.index_map[i], sims[i]) for i in highest_sims]

    def analogy(self, positives, negatives):
        """
        Computes cosine similarity between weighted average of input
        words, where the weight for positive words is 1.0, and the weight
        for negative words is -1.0. 
        """
        positives = [self.v(pos) for pos in positives]
        negatives = [-1. * self.v(neg) for neg in negatives]
        avg = np.mean(positives + negatives, axis=0)
        return self.most_similar(avg, n=1)

    def accuracy(self, questions):
        """
        Answer analogy questions like the ones found at
        https://word2vec.googlecode.com/svn/trunk/questions-words.txt
        
        Calculates accuracy for each section and for the total.

        Ignores questions that contain OOV words.
        """
        vocab = set(self.word_map.iterkeys())
        categories = []
        total = 0
        correct = 0
        skipped = 0
        with open(questions) as questions_file:
            for line in questions_file:
                if line.startswith(':'):
                    categories.append(line.split()[1])
                    continue
                else:
                    a, b, c, answer = [word.lower() for word in line.split()]
                    if any([word not in vocab for word in [a, b, c, answer]]):
                        skipped += 1
                        continue
                    predicted = self.analogy([b, c], [a])
                    print "{} {} {} <{}> :: {}".format(a, b, c, answer, predicted)
                    if predicted[0] == answer:
                        correct += 1
                    total += 1
        print "Skipped {} questions due to OOV items.".format(skipped)
        return correct / total

def setup(corpus_dir, n_files=750, min_count=10, dim=300):
    sentences = Sentences(corpus_dir, n_files=n_files)
    vocab = Vocab(sentences)
    vocab.build_vocab(min_count=min_count)
    table = UnigramTable(vocab)
    e = Embeddings(vocab, dim=dim)
    return sentences, vocab, table, e

def main(corpus_dir, n_files=100, min_count=10, dim=300):
    sentences, vocab, table, e = setup(
        corpus_dir, n_files=n_files, min_count=min_count, dim=dim)
    e.train(sentences, 0.001, table)
    e.save(syn1=True)

if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
