import string
import glob
import random
import os
from os.path import basename, dirname, join as pjoin
from nltk.tokenize import sent_tokenize

punct_map = dict((ord(char), None) for char in string.punctuation)
def strip_punct(line):
    # stackoverflow.com/questions/
    # 265960/best-way-to-strip-punctuation-from-a-string-in-python
    return line.translate(punct_map)

def preproc(indir, outdir):
    """Rewrite text files to outdir to have one sentence per line."""
    subdirs = glob.glob(indir + '/*')
    for i,subdir in enumerate(subdirs):
        print "Preprocessing directory {} of {}".format(i+1, len(subdirs))
        files = glob.glob(subdir + '/*')
        for text_file in files:
            path = dirname(text_file)
            prefix = basename(path)
            with open(text_file) as f:
                out_name = pjoin(outdir, prefix+basename(text_file))
                with open(out_name, 'w+') as out:
                    for line in f:
                        for sent in sent_tokenize(line.decode('utf-8').strip()):
                            out.write(
                                strip_punct(sent).lower().encode('utf-8')+'\n')

class Sentences(object):
    def __init__(self, dirname, n_files=None):
        self.dirname = dirname
        self.files = []
        self.word_count = 0
        for text in glob.glob(self.dirname + '/*'):
            self.files.append(text)
        if n_files is not None:
            random.seed(1234)
            random.shuffle(self.files)
            del self.files[n_files:]

    def count_words(self):
        print "Getting word count..."
        word_count = 0
        for i,text_file in enumerate(self.files):
            with open(text_file) as f:
                for line in f:
                    line = line.decode('utf-8')
                    word_count += len(line.split())
        return word_count

    def __iter__(self):
        if self.word_count == 0:
            self.word_count = self.count_words()
        file_num = 1
        for text_file in self.files:
            print "Processing file {} of {}".format(
                file_num, len(self.files))
            file_num += 1
            with open(text_file) as f:
                for line in f:
                    yield line.split()

