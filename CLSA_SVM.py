from Datasets3 import *
import argparse
import time
import embeddings
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, \
                            recall_score, confusion_matrix
from scipy.spatial.distance import cosine
import numpy as np
import warnings


from utils import *
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


def main():
    parser = argparse.ArgumentParser()
    #修改成对应的embedding地址
    #parser.add_argument('--src_vecs', default=r'./data/embedding_new/vecmap/en1010.mapped_es_vecmap.txt', help='mapped vectors of source language')
    #parser.add_argument('--trg_vecs', default=r'./data/embedding_new/vecmap/en1010.mapped_es_vecmap.txt', help='mapped vectors of target language')
    parser.add_argument('--src_vecs', default=r'./data/embedding_new/en1106.multi-fasttest.mapped_es_SLM_a.5_b.05_new.txt', help='mapped vectors of source language')
    parser.add_argument('--trg_vecs', default=r'./data/embedding_new/es1106.multi-fasttest.mapped_en_SLM_a.5_b.05_new.txt', help='mapped vectors of target language')
    #parser.add_argument('--src_vecs', default=r'./data/embedding_new/en1025.w2v.mapped_es_SLM_a.5_b.05_new.txt', help='mapped vectors of source language')
    #parser.add_argument('--trg_vecs', default=r'./data/embedding_new/en1025.w2v.mapped_es_SLM_a.5_b.05_new.txt', help='mapped vectors of target language')
    parser.add_argument('--binary', default=True, help="binary or 4-class (default: True)")
    parser.add_argument('--src_lang', default='en')
    parser.add_argument('--trg_lang', default='es')  #改成对应语言
    parser.add_argument('--dataset', default=r"opener_sents")
    args = parser.parse_args()

    dtype = "float32"
    # import datasets (representation will depend on final classifier)
    print('importing datasets')

    t1 = time.localtime()
    print("{0: 4d}-{1: 2d}-{2: 2d}-{3: 2d}:{4: 2d}:{5: 2d}".format(t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_hour,
                                                                   t1.tm_min, t1.tm_sec))
    t1 = time.time()
    # Import monolingual vectors
    print('importing word embeddings')
    src_file = open(args.src_vecs, 'r', encoding='utf-8')
    trg_file = open(args.trg_vecs, 'r', encoding='utf-8')
    src_words, src_vecs, src_num, src_dim = embeddings.read(src_file, dtype=dtype)
    trg_words, trg_vecs, trg_num, trg_dim = embeddings.read(trg_file, dtype=dtype)

    #length_norm = min(len(src_words), len(trg_words))
    src_words = src_words
    trg_words = trg_words
    src_vecs = src_vecs
    trg_vecs = trg_vecs

    src_vecs = np.asarray(src_vecs) # 每个源语言单词的向量
    trg_vecs = np.asarray(trg_vecs) # 每个目标语言的向量

    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    src_ind2word = {i: word for i, word in enumerate(src_words)}
    trg_ind2word = {i: word for i, word in enumerate(trg_words)}
    src_dict = dict(zip(src_words, src_vecs))
    trg_dict = dict(zip(trg_words, trg_vecs))


    src_dataset = General_Dataset(os.path.join(r'./datasets2', args.src_lang, args.dataset), src_vecs, src_dict,
                                      binary=True, rep=ave_vecs, one_hot=False, lowercase=False)

    trg_dataset = General_Dataset(os.path.join(r'./datasets2', args.trg_lang, args.dataset), trg_vecs, trg_dict,
                                      binary=True, rep=ave_vecs, one_hot=False, lowercase=False)

    #best_c, best_f1 = get_best_C(src_dataset, trg_dataset)
    #clf = LinearSVC(C=best_c, max_iter=5000)
    clf = SVC(kernel='rbf', verbose=True)
    clf.fit(src_dataset._Xtrain, src_dataset._ytrain)

    cpred = clf.predict(trg_dataset._Xtest)
    prec = precision_score(trg_dataset._ytest, cpred)
    rec = recall_score(trg_dataset._ytest, cpred)
    cf1 = f1_score(trg_dataset._ytest, cpred)
    acc = accuracy_score(trg_dataset._ytest, cpred)
    print('-binary-')
    #print('Acc: {0:.4f}'.format(clf.score(trg_dataset._Xtrain, trg_dataset._ytrain))) changed by Yuemei

    print('F1 Score: {0:.4f}'.format(cf1))
    print('Acc Score: {0:.4f}'.format(acc))
    print('Precision Score: {0:.4f}'.format(prec))
    print('Recall Score: {0:.4f}'.format(rec))

if __name__ == '__main__':
    main()