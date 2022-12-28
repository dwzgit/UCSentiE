import numpy as np


def ave_vecs(sentence, model, dict):
    """
    Returns an embedding which is the average
    of the word embeddings from the word embedding
    model for the tokens in a sentence.
    """
    sent = np.array(np.zeros((model.shape[1])))
    sent_length = len(sentence.split())
    for w in sentence.split():
        try:
            sent += dict[w]
        except KeyError:
            # TODO: implement a much better backoff strategy (Edit distance)
            sent += np.random.uniform(-.25, .25, model.shape[1])
    return sent / sent_length


def words(sentence, model, dict):
    """
    Returns a list of tokens. This is mainly
    for the General_Dataset class.
    """
    return sentence.split()


def getMyData(fname, label, model, dict, representation=ave_vecs, encoding='utf-8'):
    """
    Returns a list of representations (either ave_vecs or words) and label tuples
    which serve as x, y pairs for training.
    """
    data = []
    for sent in open(fname, 'r', encoding=encoding):
        data.append((representation(sent, model, dict), label))
    return data

