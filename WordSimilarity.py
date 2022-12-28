# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 08:50:13 2022

@author: xuyuemei

Calculate the similarity of CLWE 
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  
from pylab import xticks,yticks,np 


# Loading word embeddings
def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


src_path = './data/embedding_new/en1220.fasttext-uclswe.mapped_de_0.1.txt'
tgt_path = './data/embedding_new/de1220.fasttext-uclswe.mapped_en_0.1.txt'
#src_path = './data/fasttext_embeddings/wiki.multi.en.vec.txt'
#tgt_path = './data/fasttext_embeddings-process/wiki.multi.fr4.vec-process.txt'

nmax = 50000  # maximum number of word embeddings to load

src_embeddings, src_id2word, src_word2id = load_vec(src_path, nmax)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(tgt_path, nmax)

# Get nearest neighbors

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, K=5):
    print("Nearest neighbors of \"%s\":" % word)
    word2id = {v: k for k, v in src_id2word.items()}
    word_emb = src_emb[word2id[word]]
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    k_best = scores.argsort()[-K:][::-1]
    nearest_words = []
    for i, idx in enumerate(k_best):
        print('%.4f - %s' % (scores[idx], tgt_id2word[idx]))
        nearest_words.append(tgt_id2word[idx])
    return scores, nearest_words

# read binary dictionaries

file_lexicon = open('./data/MUSEdictionaries/en-de-process.txt', 'r', encoding='utf-8')
src_gold_words = []
trg_gold_words = []
binary_lexicon = {}
number_of_nearest_words = 5   # 映射的前几个单词 这里要修改代码

for line in file_lexicon.readlines():
    src_word, trg_word = line.rstrip('\n').split(' ')  #注意是\t 还是空格
    #print(src_word)
    if src_word not in binary_lexicon:
        binary_lexicon[src_word] = [trg_word]
    else:
        binary_lexicon[src_word].append(trg_word)   #读入的双语词典，一个源语言单词可能对应多个目标语言单词的映射
    #print(src_word,",",trg_word)
    src_gold_words.append(src_word)
    trg_gold_words.append(trg_word)

print(binary_lexicon)
#binary_lexicon = dict(zip(src_gold_words,trg_gold_words)) 

file_lexicon.close()

#find the nearest neighbors for each word 
count = 0
hit_count= 0
flag = 0

for key, value in binary_lexicon.items():
    src_gold_word = key
    trg_gold_words = value
    if flag <10:
        flag =flag+1
        print("词典",src_gold_word,trg_gold_words)

    # if the word in the embedding space
    if src_gold_word in src_word2id.keys():
        hit_count = hit_count+1
        #trg_gold_words = binary_lexicon.get(src_gold_word)
        similar_scores, nearest_words = get_nn(src_gold_word, src_embeddings, src_id2word, tgt_embeddings, tgt_id2word, number_of_nearest_words)

        for candidate_word in nearest_words:
            if candidate_word in trg_gold_words :
                count = count+1
                print(src_gold_word, candidate_word)
                break

#acc = count / len(src_gold_words)
#这个是在双语词典中有多少个单词被正确找到
acc1 = count / len(binary_lexicon)
print('Acc: {0:.4f}'.format(acc1))
print(f"在双语词典中有{hit_count}个单词也在构建的词典中")
print(f"双语词典和构建词典的{hit_count}个单词中，在最近邻为{number_of_nearest_words}找到了{count}个")
acc2 = count / hit_count
print('Acc: {0:.4f}'.format(acc2))
