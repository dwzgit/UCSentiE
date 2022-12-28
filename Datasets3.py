import os, re
import numpy as np
from Representations import *

class ProjectionDataset():
    """
    A wrapper for the translation dictionary. The translation dictionary
    should be word to word translations separated by a tab. The
    projection dataset only includes the translations that are found
    in both the source and target vectors.
    """
    def __init__(self, translation_dictionary, src_vecs, trg_vecs):
        (self._X, self._y, self._X_vecs, self._y_vecs) = self.getdata(translation_dictionary, src_vecs, trg_vecs)

    def train_dev_split(self, x, train=.9):  
        # split data into training and development, keeping /train/ amount for training. 将data分为train和development
        train_idx = int(len(x)*train)
        return x[:train_idx], x[train_idx:]

    def getdata(self, translation_dictionary, src_vecs, trg_vecs):
        x, y = [], []
        x_vecs, y_vecs = [], []
        for k,v in zip(translation_dictionary.keys(), translation_dictionary.values()):
            src, trg = k, v                                         #取词典中的源语言单词和目标语言单词
            if src in src_vecs.keys() and trg in trg_vecs.keys():  #若有相应词向量，将单词和对应词向量分别加入单词/词向量列表里
                x.append(src)
                x_vecs.append(src_vecs[src])
                y.append(trg)
                y_vecs.append(trg_vecs[trg])
            else:
                pass
        """xtr, xdev = self.train_dev_split(x)
        ytr, ydev = self.train_dev_split(y)"""
        return x, y, x_vecs, y_vecs   #返回相应单词表和词向量表

class General_Dataset(object):
    """This class takes as input the directory of a corpus annotated for 4 levels
    sentiment. This directory should have 4 .txt files: strneg.txt, neg.txt,
    pos.txt and strpos.txt. It also requires a word embedding model, such as
    those used in word2vec or GloVe.

    binary: instead of 4 classes you have binary (pos/neg). Default is False

    one_hot: the y labels are one hot vectors where the correct class is 1 and
             all others are 0. Default is True.

    dtype: the dtype of the np.array for each vector. Default is np.float32.

    rep: this determines how the word vectors are represented.

         ave_vecs: each sentence is represented as the average of all of the
                    word vectors in the sentence.

         idx_vecs: each sentence is respresented as a list of word ids given by
                    the word-2-idx dictionary.

         words: each sentence is a list of the tokens in the sentence
    """

    def __init__(self, DIR, model, dict, str=True, binary=False, one_hot=True,
                 dtype=np.float32, rep=ave_vecs, lowercase=True):

        self.rep = rep
        self.one_hot = one_hot
        self.lowercase = lowercase

        Xtrain, Xdev, Xtest, ytrain, ydev, ytest = self.open_data(DIR, model, binary, rep, dict)

        self._Xtrain = Xtrain
        self._ytrain = ytrain
        self._Xdev = Xdev
        self._ydev = ydev
        self._Xtest = Xtest
        self._ytest = ytest
        self._num_examples = len(self._Xtrain)

    def to_array(self, integer, num_labels):
        """quick trick to convert an integer to a one hot vector that
        corresponds to the y labels""" #将一个整数转换为一个对应于y标签的one hot向量
        integer = integer - 1
        return np.array(np.eye(num_labels)[integer])

    def open_data(self, DIR, model, binary, rep, dict):
        if binary:
            ##################
            # Binary         #
            ##################
            train_neg = getMyData(os.path.join(DIR, 'train/neg.txt'),
                                  0, model, dict, encoding='UTF-8',
                                  representation=rep)
            '''train_neg2 = getMyData(os.path.join(DIR, 'train/strneg.txt'),
                                  0, model, dict, encoding='UTF-8',
                                  representation=rep)'''
            train_pos = getMyData(os.path.join(DIR, 'train/pos.txt'),
                                  1, model, dict, encoding='UTF-8',
                                  representation=rep)
            '''train_pos2 = getMyData(os.path.join(DIR, 'train/strpos.txt'),
                                  1, model, dict, encoding='UTF-8',
                                  representation=rep)'''
            dev_neg = getMyData(os.path.join(DIR, 'dev/neg.txt'),
                                0, model, dict, encoding='UTF-8',
                                representation=rep)
            '''dev_neg2 = getMyData(os.path.join(DIR, 'dev/strneg.txt'),
                                0, model, dict, encoding='UTF-8',
                                representation=rep)'''
            dev_pos = getMyData(os.path.join(DIR, 'dev/pos.txt'),
                                1, model, dict, encoding='UTF-8',
                                representation=rep)
            '''dev_pos2 = getMyData(os.path.join(DIR, 'dev/strpos.txt'),
                                1, model, dict, encoding='UTF-8',
                                representation=rep)'''
            test_neg = getMyData(os.path.join(DIR, 'test/neg.txt'),
                                 0, model, dict, encoding='UTF-8',
                                 representation=rep)
            '''test_neg2 = getMyData(os.path.join(DIR, 'test/strneg.txt'),
                                 0, model, dict, encoding='UTF-8',
                                 representation=rep)'''
            test_pos = getMyData(os.path.join(DIR, 'test/pos.txt'),
                                 1, model, dict, encoding='UTF-8',
                                 representation=rep)
            '''test_pos2 = getMyData(os.path.join(DIR, 'test/strpos.txt'),
                                 1, model, dict, encoding='UTF-8',
                                 representation=rep)'''

            traindata = train_pos + train_neg 
            devdata = dev_pos + dev_neg 
            testdata = test_pos + test_neg

            # Set up vocab now
            self.vocab = set()

            # Training data
            Xtrain = [data for data, y in traindata]
            if self.lowercase:
                Xtrain = [[w.lower() for w in sent] for sent in Xtrain]
            if self.one_hot is True:
                ytrain = [self.to_array(y, 2) for data, y in traindata]
            else:
                ytrain = [y for data, y in traindata]
            self.vocab.update(set([w for i in Xtrain for w in i]))

            # Dev data
            Xdev = [data for data, y in devdata]
            if self.lowercase:
                Xdev = [[w.lower() for w in sent] for sent in Xdev]
            if self.one_hot is True:
                ydev = [self.to_array(y, 2) for data, y in devdata]
            else:
                ydev = [y for data, y in devdata]
            self.vocab.update(set([w for i in Xdev for w in i]))

            # Test data
            Xtest = [data for data, y in testdata]
            if self.lowercase:
                Xtest = [[w.lower() for w in sent] for sent in Xtest]
            if self.one_hot is True:
                ytest = [self.to_array(y, 2) for data, y in testdata]
            else:
                ytest = [y for data, y in testdata]
            self.vocab.update(set([w for i in Xtest for w in i]))

        else:
            ##################
            # 4 CLASS        #
            ##################
            '''train_strneg = getMyData(os.path.join(DIR, 'train/strneg.txt'),
                                  0, model, dict, encoding='latin',
                                  representation=rep)
            train_strpos = getMyData(os.path.join(DIR, 'train/strpos.txt'),
                                  3, model, dict, encoding='latin',
                                  representation=rep)'''
            train_neg = getMyData(os.path.join(DIR, 'train/neg.txt'),
                                  1, model, dict, encoding='latin',
                                  representation=rep)
            train_pos = getMyData(os.path.join(DIR, 'train/pos.txt'),
                                  2, model, dict, encoding='latin',
                                  representation=rep)
            '''dev_strneg = getMyData(os.path.join(DIR, 'dev/strneg.txt'),
                                0, model, dict, encoding='latin',
                                representation=rep)
            dev_strpos = getMyData(os.path.join(DIR, 'dev/strpos.txt'),
                                3, model, dict, encoding='latin',
                                representation=rep)'''
            dev_neg = getMyData(os.path.join(DIR, 'dev/neg.txt'),
                                1, model, dict, encoding='latin',
                                representation=rep)
            dev_pos = getMyData(os.path.join(DIR, 'dev/pos.txt'),
                                2, model, dict, encoding='latin',
                                representation=rep)
            '''test_strneg = getMyData(os.path.join(DIR, 'test/strneg.txt'),
                                 0, model, dict, encoding='latin',
                                 representation=rep)
            test_strpos = getMyData(os.path.join(DIR, 'test/strpos.txt'),
                                 3, model, dict, encoding='latin',
                                 representation=rep)'''
            test_neg = getMyData(os.path.join(DIR, 'test/neg.txt'),
                                 1, model, dict, encoding='latin',
                                 representation=rep)
            test_pos = getMyData(os.path.join(DIR, 'test/pos.txt'),
                                 2, model, dict, encoding='latin',
                                 representation=rep)

            traindata = train_pos + train_neg 
            devdata = dev_pos + dev_neg 
            testdata = test_pos + test_neg

            self.vocab = set()

            # Training data
            Xtrain = [data for data, y in traindata]
            if self.lowercase:
                Xtrain = [[w.lower() for w in sent] for sent in Xtrain]
            if self.one_hot is True:
                ytrain = [self.to_array(y, 4) for data, y in traindata]
            else:
                ytrain = [y for data, y in traindata]
            self.vocab.update(set([w for i in Xtrain for w in i]))

            # Dev data
            Xdev = [data for data, y in devdata]
            if self.lowercase:
                Xdev = [[w.lower() for w in sent] for sent in Xdev]
            if self.one_hot is True:
                ydev = [self.to_array(y, 4) for data, y in devdata]
            else:
                ydev = [y for data, y in devdata]
            self.vocab.update(set([w for i in Xdev for w in i]))

            # Test data
            Xtest = [data for data, y in testdata]
            if self.lowercase:
                Xtest = [[w.lower() for w in sent] for sent in Xtest]
            if self.one_hot is True:
                ytest = [self.to_array(y, 4) for data, y in testdata]
            else:
                ytest = [y for data, y in testdata]
            self.vocab.update(set([w for i in Xtest for w in i]))

        if self.rep is not words:
            Xtrain = np.array(Xtrain)
            Xdev = np.array(Xdev)
            Xtest = np.array(Xtest)
        ytrain = np.array(ytrain)
        ydev = np.array(ydev)
        ytest = np.array(ytest)

        return Xtrain, Xdev, Xtest, ytrain, ydev, ytest

