# -*- coding: utf-8 -*-
import sys
import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from computing_w1 import computing_w1 as cpw1


from Datasets import *
from WordVecs import *
from utils import *
from cupy_utils import *
import cupy as cp

import embeddings

from sklearn.metrics import f1_score, accuracy_score, precision_score, \
                            recall_score, confusion_matrix
from scipy.spatial.distance import cosine


class PSE(nn.Module):
    """
    Bilingual Sentiment Embeddings

    Parameters:

        src_vecs: WordVecs instance with the embeddings from the source language
        trg_vecs: WordVecs instance with the embeddings from the target language
        pdataset: Projection_Dataset from source to target language
        cdataset: Source sentiment dataset
        projection_loss: the distance metric to use for the projection loss
                         can be either mse (default) or cosine
        output_dim: the number of class labels to predict (default: 4)

    Optional:

        src_syn1: a list of positive sentiment words in the source language
        src_syn2: a second list of positive sentiment words in the
                  source language. This must be of the same length as
                  src_syn1 and should not have overlapping vocabulary
        src_neg : a list of negative sentiment words in the source language
                  this must be of the same length as src_syn1
        trg_syn1: a list of positive sentiment words in the target language
        trg_syn2: a second list of positive sentiment words in the
                  target language. This must be of the same length as
                  trg_syn1 and should not have overlapping vocabulary
        trg_neg : a list of negative sentiment words in the target language
                  this must be of the same length as trg_syn1



    """

    def __init__(self, src_vecs, src_num, src_dim, src_w2idx, src_idx2w, trg_vecs, trg_num, trg_dim, trg_w2idx, trg_idx2w, pdataset,
                 cdataset, trg_dataset,
                 projection_loss='mse',
                 output_dim=4,
                 src_syn1=None, src_syn2=None, src_neg=None,
                 trg_syn1=None, trg_syn2=None, trg_neg=None
                 ):
        super(PSE, self).__init__()

        # Embedding matrices
        self.semb = nn.Embedding(src_num, src_dim)
        self.semb.weight.data.copy_(torch.from_numpy(src_vecs))
        self.sw2idx = src_w2idx
        self.sidx2w = src_idx2w   #加载源语言预训练词嵌入向量
        self.temb = nn.Embedding(trg_num, trg_dim)
        self.temb.weight.data.copy_(torch.from_numpy(trg_vecs))
        self.tw2idx = trg_w2idx
        self.tidx2w = trg_idx2w   #加载目标语言预训练词嵌入向量

        # Projection vectors
        self.m = nn.Linear(src_dim,
                           src_dim,
                           bias=False)  #source的全连接层（即求解转换矩阵）
        self.mp = nn.Linear(trg_dim,
                            trg_dim,
                            bias=False) #target的全连接层（即求解转换矩阵）

        # Classifier
        self.clf = nn.Linear(src_dim, output_dim)  #分类器

        # Loss Functions 损失函数设置，有MSE和cos两种可选
        self.criterion = nn.CrossEntropyLoss()
        if projection_loss == 'mse':
            self.proj_criterion = mse_loss
        elif projection_loss == 'cosine':
            self.proj_criterion = cosine_loss

        # Optimizer
        self.optim = torch.optim.Adam(self.parameters())

        # Datasets
        self.pdataset = pdataset
        self.cdataset = cdataset
        self.trg_dataset = trg_dataset
        self.src_syn1 = src_syn1
        self.src_syn2 = src_syn2
        self.src_neg = src_neg
        self.trg_syn1 = trg_syn1
        self.trg_syn2 = trg_syn2
        self.trg_neg = trg_neg

        # Trg Data
        if (self.trg_dataset != None and
            self.trg_syn1 != None and
            self.trg_syn2 != None and
            self.trg_neg !=None):
            self.trg_data = True
        else:
            self.trg_data = False

        # History
        self.history = {'loss':[], 'dev_cosine':[], 'dev_f1':[], 'cross_f1':[],
                         'syn_cos':[], 'ant_cos':[], 'cross_syn':[], 'cross_ant':[]}

        # Do not update original embedding spaces
        self.semb.weight.requires_grad=False
        self.temb.weight.requires_grad=False

    def dump_weights(self):
        # Dump the weights to outfile
        w1 = self.m.weight.data.numpy()
        w2 = self.mp.weight.data.numpy()
        w3 = self.clf.weight.data.numpy()
        b = self.clf.bias.data.numpy()  #返回权重，输出到outfile中
        return w1, w2

    def load_weights(self, weight_file):
        # Load weights from weight_file
        f = np.load(weight_file)
        w1 = self.m.weight.data.copy_(torch.from_numpy(f['arr_0']))
        w2 = self.mp.weight.data.copy_(torch.from_numpy(f['arr_1']))
        w3 = self.clf.weight.data.copy_(torch.from_numpy(f['arr_2']))
        b = self.clf.bias.data.copy_(torch.from_numpy(f['arr_3']))  #将输出的权重load进来

    def project(self, X, Y):
        """
        Project X and Y into shared space.
        X is a list of source words from the projection lexicon,
        and Y is the list of single word translations.
        """
        x_nparray = cp.asnumpy(X)
        y_nparray = cp.asnumpy(Y)
        x_tensor = torch.from_numpy(x_nparray)
        y_tensor = torch.from_numpy(y_nparray)
        x_proj = self.m(x_tensor)
        y_proj = self.mp(y_tensor)
        return x_proj, y_proj

    def project_one(self, x, src=True):
        """
        Project only a single list of words to the shared space.
        只讲source或target转换到共享空间中
        """
        if src:
            x_nparray = cp.asnumpy(x)
            x_tensor = torch.from_numpy(x_nparray)
            x_proj = self.m(x_tensor)
        else:
            x_nparray = cp.asnumpy(x)
            x_tensor = torch.from_numpy(x_nparray)
            x_proj = self.mp(x_tensor)
        return x_proj

    def projection_loss(self, x, y):
        """
        Find the loss between the two projected sets of translations.
        The loss is the proj_criterion.
        计算映射后的loss
        """

        x_proj = self.project_one(x, src=True)

        # distance-based loss (cosine, mse)
        y_nparray = cp.asnumpy(y)
        y_tensor = torch.from_numpy(y_nparray)
        loss = self.proj_criterion(x_proj, y_tensor)

        return loss, x_proj

    def idx_vecs(self, sentence, model):
        """
        Converts a tokenized sentence to a vector
        of word indices based on the model.
        把一句话转换成对应单词序号的向量
        """
        sent = []
        for w in sentence:
            try:
                sent.append(model[w])
            except KeyError:
                sent.append(0)
        return torch.LongTensor(np.array(sent))

    def lookup(self, X, model):
        """
        Converts a batch of tokenized sentences
        to a matrix of word indices from model.
        把一些带标记的句子转换成词序号的矩阵
        """
        return [self.idx_vecs(s, model) for s in X]

    def ave_vecs(self, X, src=True):
        """
        Converts a batch of tokenized sentences into
        a matrix of averaged word embeddings. If src
        is True, it uses the sw2idx and semb to create
        the averaged representation. Otherwise, it uses
        the tw2idx and temb.
        把一些样本转换成对应词嵌入向量加和的平均，生成一个矩阵，如果src=True，则
        """
        vecs = []
        #import pdb; pdb.set_trace()
        if src:
            # idxs = np.array(self.lookup(X, self.sw2idx))
            idxs = self.lookup(X, self.sw2idx)
            for i in idxs:
                vecs.append(self.semb(Variable(i)).mean(0))
        else:
            # idxs = np.array(self.lookup(X, self.tw2idx))
            idxs = self.lookup(X, self.tw2idx)
            for i in idxs:
                vecs.append(self.temb(Variable(i)).mean(0))
        return torch.stack(vecs)

    def ave_vecs_new(self, X, trg_vecs, src=True):
        vecs = []
        if src:
            idxs = self.lookup(X, self.sw2idx)
            for i in idxs:
                vecs.append(self.semb(Variable(i)).mean(0))
        else:
            idxs = self.lookup(X, self.tw2idx)
            trg_vecs_np = cp.asnumpy(trg_vecs)
            self.temb.weight.data.copy_(torch.from_numpy(trg_vecs_np))
            for i in idxs:
                vecs.append(self.temb(Variable(i)).mean(0))
        return torch.stack(vecs)

    def predict(self, X, src=True):
        """
        Projects the averaged embeddings from X
        to the joint space and then uses either
        m (if src==True) or mp (if src==False)
        to predict the sentiment of X.
        先映射，然后使用m或者mp进行情感预测
        """

        X = self.ave_vecs(X, src)
        if src:
            x_proj = self.m(X)
        else:
            x_proj = self.mp(X)
        out = F.softmax(self.clf(x_proj))
        return out

    def predict_new(self, X, trg_vecs, src=True):
        X = self.ave_vecs_new(X, trg_vecs, src=src)
        out = F.softmax(self.clf(X))
        return out

    def predict_labels(self, X, src=True):
        label_dict = {2: {0: "negative",
                          1: "positive"},
                      4: {0: "strong negative",
                          1: "negative",
                          2: "positive",
                          3: "strong positive"}
                      }
        out = self.predict(X, src=True)
        preds = out.argmax(dim=1)
        output_dim = self.clf.weight.shape[0]
        labels = [label_dict[output_dim][l] for l in preds.tolist()]
        return labels


    def classification_loss(self, x, y, src=True):
        pred = self.predict(x, src=src)
        y = Variable(torch.from_numpy(y))
        y = y.long()
        loss = self.criterion(pred, y)
        return loss

    def full_loss(self, proj_x, proj_y, class_x, class_y,
                  alpha=.5):
        """
        Calculates the combined projection and classification loss.
        Alpha controls the amount of weight given to each loss term.
        alpha是参数，控制两个loss所占比例
        """

        proj_loss, x_proj = self.projection_loss(proj_x, proj_y)
        class_loss = self.classification_loss(class_x, class_y, src=True)
        return alpha * proj_loss + (1 - alpha) * class_loss, x_proj

    def dropout(m, p):
        if p <= 0.0:
            return m
        else:
            xp = get_array_module(m)
            mask = xp.random.rand(*m.shape) >= p
            return m * mask

    def fit(self, xp, proj_X, proj_Y,
            class_X, class_Y,
            num_epochs,
            epochs,
            weight_dir='savedir',
            batch_size=100,
            alpha=0.001,
            ):

        """
        Trains the model on the projection data (and
        source language sentiment data (class_X, class_Y).
        训练模型
        """

        num_batches = int(len(class_X) / batch_size)
        for i in range(epochs):
            idx = 0
            for j in range(num_batches):
                cx = class_X[idx:idx+batch_size]
                cy = class_Y[idx:idx+batch_size]
                idx += batch_size
                self.optim.zero_grad()
                loss, x_proj = self.full_loss(proj_X, proj_Y, cx, cy, alpha)
                loss.backward()
                self.optim.step()

        """
            # check cosine distance between dev translation pairs
            xdev = self.pdataset._Xdev
            ydev = self.pdataset._ydev
            xp, yp = self.project(xdev, ydev)
            score = cos(xp, yp)

            # check source dev f1
            xdev = self.cdataset._Xdev
            ydev = self.cdataset._ydev
            xp = self.predict(xdev).data.numpy().argmax(1)
            # macro f1
            dev_f1 = macro_f1(ydev, xp)

            # check cosine distance between source sentiment synonyms
            p1 = self.project_one(self.src_syn1)
            p2 = self.project_one(self.src_syn2)
            syn_cos = cos(p1, p2)

            # check cosine distance between source sentiment antonyms
            p3 = self.project_one(self.src_syn1)
            n1 = self.project_one(self.src_neg)
            ant_cos = cos(p3, n1)

        # If there's no target data
        if not self.trg_data:

            if dev_f1 > best_cross_f1:
                best_cross_f1 = dev_f1
                weight_file = r'./savedir/' + '{0}epochs-{1}batchsize-{2}alpha-{3:.3f}devf1'.format(num_epochs, batch_size1, alpha, best_cross_f1)
                self.dump_weights(weight_file)
            sys.stdout.write('\r epoch {0} loss: {1:.3f}  trans: {2:.3f}  src_f1: {3:.3f}  src_syn: {4:.3f}  src_ant: {5:.3f}'.format(num_epochs, loss.data.item(), score.data.item(), dev_f1, syn_cos.data.item(), ant_cos.data.item()))
            sys.stdout.flush()
            self.history['loss'].append(loss.data.item())
            self.history['dev_cosine'].append(score.data.item())
            self.history['dev_f1'].append(dev_f1)
            self.history['syn_cos'].append(syn_cos.data.item())
            self.history['ant_cos'].append(ant_cos.data.item())

        # If there's target data
        elif self.trg_data:
            # check target dev f1
            crossx = self.trg_dataset._Xdev
            crossy = self.trg_dataset._ydev
            xp = self.predict(crossx, src=False).data.numpy().argmax(1)
            # macro f1
            cross_f1 = macro_f1(crossy, xp)


            if cross_f1 > best_cross_f1:
                best_cross_f1 = cross_f1
                weight_file = r'./savedir/' + '{0}epochs-{1}batchsize-{2}alpha-{3:.3f}devf1'.format(num_epochs, batch_size1, alpha, best_cross_f1)
                self.dump_weights(weight_file)

            # check cosine distance between target sentiment synonyms
            cp1 = self.project_one(self.trg_syn1, src=False)
            cp2 = self.project_one(self.trg_syn2, src=False)
            cross_syn_cos = cos(cp1, cp2)

            # check cosine distance between target sentiment antonyms
            cp3 = self.project_one(self.trg_syn1, src=False)
            cn1 = self.project_one(self.trg_neg, src=False)
            cross_ant_cos = cos(cp3, cn1)
            sys.stdout.write('\r epoch {0} loss: {1:.3f}  trans: {2:.3f}  src_f1: {3:.3f}  trg_f1: {4:.3f}  src_syn: {5:.3f}  src_ant: {6:.3f}  cross_syn: {7:.3f}  cross_ant: {8:.3f}'.format(num_epochs, loss.data[0], score.data[0], dev_f1,
                cross_f1, syn_cos.data[0], ant_cos.data[0],
                cross_syn_cos.data[0], cross_ant_cos.data[0]))
            sys.stdout.flush()
            self.history['loss'].append(loss.data[0])
            self.history['dev_cosine'].append(score.data[0])
            self.history['dev_f1'].append(dev_f1)
            self.history['cross_f1'].append(cross_f1)
            self.history['syn_cos'].append(syn_cos.data[0])
            self.history['ant_cos'].append(ant_cos.data[0])
            self.history['cross_syn'].append(cross_syn_cos.data[0])
            self.history['cross_ant'].append(cross_ant_cos.data[0])"""
        x_nparray = x_proj.detach().numpy()
        x_cparray = cp.asarray(x_nparray)
        w1, w2 = self.dump_weights()
        w1 = cp.asarray(w1)
        return x_cparray, proj_Y, w1

    def plot(self, title=None, outfile=None):
        """
        Plots the progression of the model. If outfile != None,
        the plot is saved to outfile.
        """

        h = self.history
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(h['dev_cosine'], label='translation_cosine')
        ax.plot(h['dev_f1'], label='source_f1', linestyle=':')
        ax.plot(h['cross_f1'], label='target_f1', linestyle=':')
        ax.plot(h['syn_cos'], label='source_synonyms', linestyle='--')
        ax.plot(h['ant_cos'], label='source_antonyms', linestyle='-.')
        ax.plot(h['cross_syn'], label='target_synonyms', linestyle='--')
        ax.plot(h['cross_ant'], label='target_antonyms', linestyle='-.')
        ax.set_ylim(-.5, 1.4)
        ax.legend(
                loc='upper center', bbox_to_anchor=(.5, 1.05),
                ncol=3, fancybox=True, shadow=True)
        if title:
            ax.title(title)
        if outfile:
            plt.savefig(outfile)
        else:
            plt.show()

    def confusion_Matrix(self, X, Y, src=True):
        """
        Prints a confusion matrix for the model
        """
        pred = self.predict(X, src=src).data.numpy().argmax(1)
        cm = confusion_matrix(Y, pred)
        print(cm)

    def evaluate_new(self, X, Y, trg_vecs, src=False):
        pred = self.predict_new(X, trg_vecs, src=src).data.numpy().argmax(1)
        acc = accuracy_score(Y, pred)
        prec = per_class_prec(Y, pred).mean()
        rec = per_class_rec(Y, pred).mean()
        f1 = macro_f1(Y, pred)
        print('Test Set:')
        print('acc:  {0:.3f}\nmacro prec: {1:.3f}\nmacro rec: {2:.3f}\nmacro f1: {3:.3f}'.format(acc, prec, rec, f1))

    def evaluate(self, X, Y, src=True, outfile=None):
        """
        Prints the accuracy, macro precision, macro recall,
        and macro F1 of the model on X. If outfile != None,
        the predictions are written to outfile.
        """

        pred = self.predict(X, src=src).data.numpy().argmax(1)
        acc = accuracy_score(Y, pred)
        prec = per_class_prec(Y, pred).mean()
        rec = per_class_rec(Y, pred).mean()
        f1 = macro_f1(Y, pred)
        if outfile:
            with open(outfile, 'w') as out:
                for i in pred:
                    out.write('{0}\n'.format(i))
        else:
            print('Test Set:')
            print('acc:  {0:.3f}\nmacro prec: {1:.3f}\nmacro rec: {2:.3f}\nmacro f1: {3:.3f}'.format(acc, prec, rec, f1))


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        xp = get_array_module(m)
        mask = xp.random.rand(*m.shape) >= p
        return m * mask

def init_dict(xp, x, z, src_words, trg_words, init_dir, unsupervised_vocab):
    normalize = ['unit', 'center', 'unit']
    embeddings.normalize(x, normalize)
    embeddings.normalize(z, normalize)

    # Build the seed dictionary
    direction = 'union'
    csls_neighborhood = 10
    src_indices = []
    trg_indices = []

    # initialization
    sim_size = min(x.shape[0], z.shape[0]) if unsupervised_vocab <= 0 else min(x.shape[0], z.shape[0], unsupervised_vocab)  # 保持X矩阵和Z矩阵行数相等
    x = xp.asarray(x)
    z = xp.asarray(z)
    u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)  # 对于X矩阵进行奇异值分解
    xsim = (u * s).dot(u.T)
    u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)  # 对于Z矩阵进行奇异值分解
    zsim = (u * s).dot(u.T)
    del u, s, vt
    # 分别对Xsim和Zsim进行排序得到sortX和sortZ
    xsim.sort(axis=1)
    zsim.sort(axis=1)
    # 对上述两个矩阵进行标准化
    embeddings.normalize(xsim, normalize)
    embeddings.normalize(zsim, normalize)
    sim = xsim.dot(zsim.T)
    if csls_neighborhood > 0:
        knn_sim_fwd = topk_mean(sim, k=csls_neighborhood)
        knn_sim_bwd = topk_mean(sim.T, k=csls_neighborhood)
        sim -= knn_sim_fwd[:, xp.newaxis] / 2 + knn_sim_bwd / 2
    if direction == 'forward':
        src_indices = xp.arange(sim_size)
        trg_indices = sim.argmax(axis=1)
    elif direction == 'backward':
        src_indices = sim.argmax(axis=0)
        trg_indices = xp.arange(sim_size)
    elif direction == 'union':
        src_indices = xp.concatenate((xp.arange(sim_size), sim.argmax(axis=0)))
        trg_indices = xp.concatenate((sim.argmax(axis=1), xp.arange(sim_size)))
    del xsim, zsim, sim

    ls_src_indices = src_indices.tolist()
    ls_trg_indices = trg_indices.tolist()

    D = dict()
    for i, j in zip(ls_src_indices, ls_trg_indices):
        D[src_words[i]] = trg_words[j]

    # Write initial solution
    init_sol = open(init_dir, mode='w', encoding='utf-8', errors='surrogateescape')
    for i, j in zip(ls_src_indices, ls_trg_indices):
        init_sol.write(src_words[i] + '\t' + trg_words[j] + '\n')
    init_sol.close()

    return trg_indices, src_indices

def mse_loss(x,y):
    # mean squared error loss
    #均方误差
    return torch.sum((x - y)**2) / x.data.shape[0]

def cosine_loss(x,y):
    c = nn.CosineSimilarity()
    return (1 - c(x,y)).mean()

def cos(x, y):
    """
    This returns the mean cosine similarity between two sets of vectors.
    平均余弦相似度
    """
    c = nn.CosineSimilarity()
    return c(x, y).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sl', '--source_lang',
                        help="source language",
                        default='en')   #源语言
    parser.add_argument('-tl', '--target_lang',
                        help="target language",
                        default='de')   #目标语言  这里要根据语言修改！！！！
    parser.add_argument('-bi', '--binary',
                        help="binary or 4-class (default: True)",
                        default=True,
                        type=str2bool)      #分类为二分类或四分类
    parser.add_argument('-e', '--epochs',
                        help="training epochs (default: 200)",
                        default=200,
                        type=int)   #训练周期
    parser.add_argument('-a', '--alpha',
                        help="trade-off between projection and classification objectives (default: .001)",
                        default=.5,
                        type=float)     #平衡权重
    parser.add_argument('-pl', '--proj_loss',
                        help="projection loss: mse, cosine (default: mse)",
                        default='cosine')  #映射损失
    parser.add_argument('-bs', '--batch_size',
                        help="classification batch size (default: 50)",
                        default=5000,
                        type=int)   #样本数
    parser.add_argument('-sv', '--src_vecs',
                        help=" source language vectors (default: GoogleNewsVecs )",
                        default=r'./data/fasttext_embeddings/wiki.multi.en.vec.txt')   #源语言词向量
    parser.add_argument('-tv', '--trg_vecs',
                        help=" target language vectors (default: SGNS on Wikipedia)",
                        default=r'./data/fasttext_embeddings/wiki.multi.de.vec.txt')    #目标语言词向量
    parser.add_argument('-da', '--dataset',
                        help="dataset to train and test on (default: opener_sents)",
                        default='opener_sents',)    #训练和测试数据集
    parser.add_argument('-sd', '--savedir',
                        help="where to dump weights during training (default: ./models)",
                        default=r'./savedir')  #存储位置
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda (requires cupy)')
    parser.add_argument('--csls', type=int, nargs='?', default=10, metavar='NEIGHBORHOOD_SIZE', dest='csls_neighborhood', help='use CSLS for dictionary induction')
    parser.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='the direction for dictionary induction (defaults to union)')
    parser.add_argument('--stochastic_interval', default=30, type=int,help='stochastic dictionary induction interval (defaults to 50)')
    parser.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*',
                               default=['unit', 'center', 'unit'], help='the normalization actions to perform in order')
    parser.add_argument('--vocabulary_cutoff', type=int, default=20000,
                                     help='restrict the vocabulary to the top k entries')
    parser.add_argument('--src_out', default=r'./data/embedding_new/en1220.fasttext-uclswe.mapped_de_0.3.txt')
    parser.add_argument('--trg_out', default=r'./data/embedding_new/de1220.fasttext-uclswe.mapped_en_0.3.txt')
    parser.add_argument('--lexicon', default=r'./lexicons/lexicons_inducted/en_de-uclswe-0.3-1220.txt', help="the lexicon inducted by self_learning_model")
    parser.add_argument('--log', default=r'./log/log1220-en-de-uclswe0.3.txt', help="output log")
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                                     help='write log information to stderr at each iteration')
    parser.add_argument('--validation', default=None, metavar='DICTIONARY', help='a dictionary file for validation at each iteration')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--threshold', default=0.000001, type=float,
                                     help='the convergence threshold (defaults to 0.000001)')
    parser.add_argument('--stochastic_multiplier', default=2.0, type=float, help='stochastic dictionary induction multiplier (defaults to 2.0)')
    parser.add_argument('--beta', type=float, default=.3)
    parser.add_argument('--whiten', action='store_true', help='whiten the embeddings')
    parser.add_argument('--src_reweight', type=float, default=0, nargs='?', const=1,
                               help='re-weight the source language embeddings')
    parser.add_argument('--trg_reweight', type=float, default=0, nargs='?', const=1,
                               help='re-weight the target language embeddings')
    parser.add_argument('--src_dewhiten', choices=['src', 'trg'],
                               help='de-whiten the source language embeddings')
    parser.add_argument('--trg_dewhiten', choices=['src', 'trg'],
                               help='de-whiten the target language embeddings')
    parser.add_argument('--dim_reduction', type=int, default=0, help='apply dimensionality reduction')
    parser.add_argument('--init_vocab', type=int, default=4000)
    args = parser.parse_args()

    parser.set_defaults(normalize=['unit', 'center', 'unit'],
                        whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg',
                        vocabulary_cutoff=20000, csls_neighborhood=10)
    dtype = "float32"
    # import datasets (representation will depend on final classifier)
    print('importing datasets')

    t1 = time.localtime()
    print("{0: 4d}-{1: 2d}-{2: 2d}-{3: 2d}:{4: 2d}:{5: 2d}".format(t1.tm_year, t1.tm_mon, t1.tm_mday, t1.tm_hour, t1.tm_min, t1.tm_sec))
    t1 = time.time()
    # Import monolingual vectors
    print('importing word embeddings')
    src_file = open(args.src_vecs, 'r', encoding='utf-8')
    trg_file = open(args.trg_vecs, 'r', encoding='utf-8')
    src_words, src_vecs, src_num, src_dim = embeddings.read(src_file, dtype=dtype)
    trg_words, trg_vecs, trg_num, trg_dim = embeddings.read(trg_file, dtype=dtype)
    src_dict = dict(zip(src_words, src_vecs))
    trg_dict = dict(zip(trg_words, trg_vecs))

    src_vecs = np.asarray(src_vecs, dtype='float32')
    trg_vecs = np.asarray(trg_vecs, dtype='float32')

    dataset = General_Dataset(r"./datasets/" + args.source_lang + r"/" + args.dataset,
                              src_dict, src_dict,
                              binary=args.binary,
                              rep=words,
                              one_hot=False)

    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}
    src_ind2word = {i: word for i, word in enumerate(src_words)}
    trg_ind2word = {i: word for i, word in enumerate(trg_words)}

    embeddings.normalize(src_vecs, args.normalize)
    embeddings.normalize(trg_vecs, args.normalize)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        print("cuda is not available.")

    # Get sentiment synonyms and antonyms to check how they move during training
    synonyms1, synonyms2, neg = get_syn_ant(args.source_lang, src_word2ind)


    # Use cuda
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = cp
    else:
        xp = np
    xp.random.seed(0)

    init_dir = r'./init_dict' + '_' + args.source_lang + '_' + args.target_lang + '.txt'
    init_dir_trans = r'./data/dictionaries/en-de-uclswe0.3-1220.txt'

    init_words = []
    init_src_vecs = []
    with open(r"./data/dictionaries/en-senti.txt", 'r', encoding='utf-8') as f:  #源语言带标注数据集
        for line in f.readlines():
            word = line.rstrip('\n')
            if word in src_word2ind.keys():
                idx = src_word2ind[word]
                vec = src_vecs[idx]
                init_words.append(word)
                init_src_vecs.append(vec)
            else:
                pass

    init_src_vecs = xp.array(init_src_vecs)
    # 使用的是源语言的情感词，种子词的个数是0，init_dir_trans是保存的初始解
    trg_indices_init, src_indices_init = init_dict(xp, init_src_vecs, trg_vecs, init_words, trg_words, init_dir_trans, unsupervised_vocab=0)
    # 这里使用的是单语言Embedding中的英语词和目标语言单词，种子词个数是4000，init_dir是保存的初始解
    trg_indices, src_indices = init_dict(xp, src_vecs, trg_vecs, src_words, trg_words, init_dir, args.init_vocab)

    init_trans_dict = {}
    with open(init_dir_trans, 'r', encoding='utf-8') as f:  #读初始解
        for line in f.readlines():
            src, trg = line.rstrip("\n").split('\t')
            init_trans_dict[src] = trg

    pdataset = ProjectionDataset(init_trans_dict, src_dict, trg_dict)

    if args.binary:
        output_dim = 2
        b = 'bi'
    else:
        output_dim = 4
        b = '4cls'

    pse = PSE(src_vecs, src_num, src_dim, src_word2ind, src_ind2word, trg_vecs, trg_num, trg_dim, trg_word2ind, trg_ind2word, pdataset, dataset, None,
                projection_loss=args.proj_loss,
                output_dim=output_dim,
                src_syn1=synonyms1, src_syn2=synonyms2, src_neg=neg)

    # If there's no savedir, create it
    os.makedirs(args.savedir, exist_ok=True)

    dtype = "float32"

    x = xp.asarray(src_vecs)
    y = xp.asarray(trg_vecs)
    init_x = pdataset._X_vecs
    init_y = pdataset._y_vecs

    if args.log:
        log = open(args.log, mode='w', encoding=args.encoding, errors='surrogateescape')

    best_cross_f1 = 0
    num_epochs = 0
    # Allocate memory

    x_w = xp.empty_like(x)
    y_w = xp.empty_like(y)
    # Training loop
    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    keep_prob = 0.1
    t = time.time()
    end = False
    src_size = x.shape[0] if args.vocabulary_cutoff <= 0 else min(x.shape[0], args.vocabulary_cutoff)
    trg_size = y.shape[0] if args.vocabulary_cutoff <= 0 else min(y.shape[0], args.vocabulary_cutoff)
    simfwd = xp.empty((args.batch_size, trg_size), dtype=dtype)
    simbwd = xp.empty((args.batch_size, src_size), dtype=dtype)
    knn_sim_fwd = xp.zeros(src_size, dtype=dtype)
    knn_sim_bwd = xp.zeros(trg_size, dtype=dtype)
    best_sim_forward = xp.full(src_size, -100, dtype=dtype)
    src_indices_forward = xp.arange(src_size)
    trg_indices_forward = xp.zeros(src_size, dtype=int)
    best_sim_backward = xp.full(trg_size, -100, dtype=dtype)
    src_indices_backward = xp.zeros(trg_size, dtype=int)
    trg_indices_backward = xp.arange(trg_size)
    # Fit model
    x_proj1, y_proj1, w1 = pse.fit(xp, init_x, init_y, dataset._Xtrain, dataset._ytrain, num_epochs=num_epochs, epochs=200, weight_dir=args.savedir, alpha=args.alpha)
    y_proj1 = xp.array(y_proj1)
    w1 = cpw1(x_proj1, y_proj1, direction=args.direction)


    while True:
        num_epochs += 1
        # Increase the keep probability if we have not improve in args.stochastic_interval iterations
        if it - last_improvement > args.stochastic_interval:
            if keep_prob >= 1.0:
                end = True
            keep_prob = min(1.0, args.stochastic_multiplier * keep_prob)
            last_improvement = it
        if not end:
            u, s, vt = xp.linalg.svd(y[trg_indices].T.dot(x[src_indices]))
            w3 = vt.T.dot(u.T)

            """if num_epochs > 1:
                x_proj1, y_proj1, w1 = pse.fit(xp, x_w[src_indices], y_w[trg_indices], dataset._Xtrain, dataset._ytrain,
                                                num_epochs=num_epochs, epochs=10, weight_dir=args.savedir,
                                                alpha=args.alpha)
                y_proj1 = xp.array(y_proj1)
                w1 = cpw1(x_proj1, y_proj1, direction=args.direction)"""

            xw = w3 * (1 - args.beta) + w1 * args.beta
            x_w = xp.asarray(x.dot(xw).get())
            y_w = y
        else:
             # advanced mapping

            # TODO xw.dot(wx2, out=xw) and alike not working
            x_w[:] = x
            y_w[:] = y

            # STEP 1: Whitening
            def whitening_transformation(m):
                u, s, vt = xp.linalg.svd(m, full_matrices=False)
                return vt.T.dot(xp.diag(1 / s)).dot(vt)

            if args.whiten:
                wx1 = whitening_transformation(x_w[src_indices])
                wy1 = whitening_transformation(y_w[trg_indices])
                x_w = x_w.dot(wx1)
                y_w = y_w.dot(wy1)

            # STEP 2: Orthogonal mapping
            wx2, s, wy2_t = xp.linalg.svd(x_w[src_indices].T.dot(y_w[trg_indices]))
            wy2 = wy2_t.T
            x_w = x_w.dot(wx2)
            y_w = y_w.dot(wy2)

            # STEP 3: Re-weighting
            x_w *= s ** args.src_reweight
            y_w *= s ** args.trg_reweight

            # STEP 4: De-whitening
            if args.src_dewhiten == 'src':
                x_w = x_w.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.src_dewhiten == 'trg':
                x_w = x_w.dot(wy2.T.dot(xp.linalg.inv(wy1)).dot(wy2))
            if args.trg_dewhiten == 'src':
                y_w = y_w.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.trg_dewhiten == 'trg':
                y_w = y_w.dot(wy2.T.dot(xp.linalg.inv(wy1)).dot(wy2))

            # STEP 5: Dimensionality reduction
            if args.dim_reduction > 0:
                x_w = x_w[:, :args.dim_reduction]
                y_w = y_w[:, :args.dim_reduction]


        if end:
            break
        else:
            # Update the training dictionary
            if args.direction in ('forward', 'union'):
                if args.csls_neighborhood > 0:
                    for i in range(0, trg_size, simbwd.shape[0]):
                        j = min(i + simbwd.shape[0], trg_size)
                        simbwd[:j-i] = xp.dot(y_w[i:j], x_w[:src_size].T)
                        knn_sim_bwd[i:j] = topk_mean(simbwd[:j-i], k=args.csls_neighborhood, inplace=True)
                for i in range(0, src_size, simfwd.shape[0]):
                    j = min(i + simfwd.shape[0], src_size)
                    simfwd[:j-i] = xp.dot(x_w[i:j], y_w[:trg_size].T)
                    simfwd[:j-i].max(axis=1, out=best_sim_forward[i:j])
                    simfwd[:j-i] -= knn_sim_bwd / 2  # Equivalent to the real CSLS scores for NN
                    dropout(simfwd[:j-i], 1 - keep_prob).argmax(axis=1, out=trg_indices_forward[i:j])
            if args.direction in ('backward', 'union'):
                if args.csls_neighborhood > 0:
                    for i in range(0, src_size, simfwd.shape[0]):
                        j = min(i + simfwd.shape[0], src_size)
                        simfwd[:j-i] = xp.dot(x_w[i:j], y_w[:trg_size].T)
                        knn_sim_fwd[i:j] = topk_mean(simfwd[:j-i], k=args.csls_neighborhood, inplace=True)
                for i in range(0, trg_size, simbwd.shape[0]):
                    j = min(i + simbwd.shape[0], trg_size)
                    simbwd[:j-i] = xp.dot(y_w[i:j], x_w[:src_size].T)
                    simbwd[:j-i].max(axis=1, out=best_sim_backward[i:j])
                    simbwd[:j-i] -= knn_sim_fwd / 2  # Equivalent to the real CSLS scores for NN
                    dropout(simbwd[:j-i], 1 - keep_prob).argmax(axis=1, out=src_indices_backward[i:j])
            if args.direction == 'forward':
                src_indices = src_indices_forward
                trg_indices = trg_indices_forward
            elif args.direction == 'backward':
                src_indices = src_indices_backward
                trg_indices = trg_indices_backward
            elif args.direction == 'union':
                src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
                trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))

            if args.direction == 'forward':
                objective = xp.mean(best_sim_forward).tolist()
            elif args.direction == 'backward':
                objective = xp.mean(best_sim_backward).tolist()
            elif args.direction == 'union':
                objective = (xp.mean(best_sim_forward) + xp.mean(best_sim_backward)).tolist() / 2
            if objective - best_objective >= args.threshold:
                last_improvement = it
                best_objective = objective

            ls_src_indices = src_indices.tolist()
            ls_trg_indices = trg_indices.tolist()
            src_vecs_new = []
            trg_vecs_new = []
            for i, j in zip(ls_src_indices, ls_trg_indices):
                src_vecs_new.append(src_dict[src_ind2word[i]])
                trg_vecs_new.append(trg_dict[trg_ind2word[j]])
            src_vecs_new = xp.array(src_vecs_new)
            trg_vecs_new = xp.array(trg_vecs_new)

            # Logging
            duration = time.time() - t
            if args.verbose:
                print(file=sys.stderr)
                print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
                print('\t- Objective:        {0:9.4f}%'.format(100 * objective), file=sys.stderr)
                print('\t- Drop probability: {0:9.4f}%'.format(100 - 100 * keep_prob), file=sys.stderr)
                sys.stderr.flush()

        t = time.time()
        it += 1

    t2 = time.localtime()
    print("{0: 4d}-{1: 2d}-{2: 2d}-{3: 2d}:{4: 2d}:{5: 2d}".format(t2.tm_year, t2.tm_mon, t2.tm_mday, t2.tm_hour, t2.tm_min, t2.tm_sec))
    t3 = time.time()-t1
    print("{0: .2f}s".format(t3))
    srcfile = open(args.src_out, mode='w', encoding='utf-8', errors='surrogateescape')
    trgfile = open(args.trg_out, mode='w', encoding='utf-8', errors='surrogateescape')
    embeddings.write(src_words, x_w, srcfile)
    embeddings.write(trg_words, y_w, trgfile)
    srcfile.close()
    trgfile.close()

    # Get best dev f1 and weights
    #best_f1, best_params, best_weights = get_best_run(args.savedir)
    #pse.load_weights(best_weights)
    #print() 
    """
    print('Dev set')
    print('best dev f1: {0:.3f}'.format(best_f1))"""
    #print('best objective: {:.4f}'.format(best_objective))

    #file = open(r'./best_obj.txt', 'w', encoding='utf-8')
    #file.write(str(best_objective))
    #file.close()
    




if __name__ == '__main__':
    main()