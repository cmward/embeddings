import string
import glob
import random
from nltk.tokenize import sent_tokenize

punct_map = dict((ord(char), None) for char in string.punctuation)
def strip_punct(line):
    # stackoverflow.com/questions/
    # 265960/best-way-to-strip-punctuation-from-a-string-in-python
    return line.translate(punct_map)

class Sentences(object):
    def __init__(self, dirname, n_files=None):
        self.dirname = dirname
        self.subdirs = glob.glob(self.dirname + '/*')
        self.files = []
        self.word_count = 0
        for subdir in self.subdirs:
            self.files.extend(glob.glob(subdir + '/*'))
        if n_files is not None:
            random.seed(1)
            random.shuffle(self.files)
            del self.files[n_files:]

    def count_words(self):
        print "Getting word count..."
        word_count = 0
        for i,text_file in enumerate(self.files):
            with open(text_file) as f:
                for line in f:
                    line = line.decode('utf-8')
                    word_count += len(strip_punct(line).lower().split())
        return word_count

    def __iter__(self):
        if self.word_count == 0:
            self.word_count = self.count_words()
        file_num = 1
        for text_file in self.files:
            print "Processing file {0} of {1}".format(
                file_num, len(self.files))
            file_num += 1
            with open(text_file) as f:
                for line in f:
                    for sent in sent_tokenize(line.decode('utf-8')):
                        yield strip_punct(sent).lower().split()

