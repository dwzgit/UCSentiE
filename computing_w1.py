import cupy as cp
from cupy_utils import *

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

def computing_w1(x_proj, y_proj, direction='union'):
    dtype = 'float32'
    src_size = min(x_proj.shape[0], 20000)
    trg_size = min(y_proj.shape[0], 20000)
    xp = cp
    simfwd = xp.empty((5000, src_size), dtype=dtype)
    simbwd = xp.empty((5000, trg_size), dtype=dtype)
    knn_sim_fwd = xp.zeros(src_size, dtype=dtype)
    knn_sim_bwd = xp.zeros(trg_size, dtype=dtype)
    csls_neighborhood = 10
    best_sim_forward = xp.full(src_size, -100, dtype=dtype)
    best_sim_backward = xp.full(trg_size, -100, dtype=dtype)
    src_indices_forward = xp.arange(src_size)
    trg_indices_forward = xp.zeros(src_size, dtype=int)
    src_indices_backward = xp.zeros(trg_size, dtype=int)
    trg_indices_backward = xp.arange(trg_size)

    for i in range(0, trg_size, simbwd.shape[0]):
        j = min(i + simbwd.shape[0], trg_size)
        simbwd[:j - i] = xp.dot(y_proj[i:j], x_proj[:src_size].T)
        knn_sim_bwd[i:j] = topk_mean(simbwd[:j - i], k=csls_neighborhood, inplace=True)
    for i in range(0, src_size, simfwd.shape[0]):
        j = min(i + simfwd.shape[0], src_size)
        simfwd[:j - i] = xp.dot(x_proj[i:j], y_proj[:trg_size].T)
        simfwd[:j - i].max(axis=1, out=best_sim_forward[i:j])
        simfwd[:j - i] -= knn_sim_bwd / 2  # Equivalent to the real CSLS scores for NN
    for i in range(0, src_size, simfwd.shape[0]):
        j = min(i + simfwd.shape[0], src_size)
        simfwd[:j - i] = xp.dot(x_proj[i:j], y_proj[:trg_size].T)
        knn_sim_fwd[i:j] = topk_mean(simfwd[:j - i], k=csls_neighborhood, inplace=True)
    for i in range(0, trg_size, simbwd.shape[0]):
        j = min(i + simbwd.shape[0], trg_size)
        simbwd[:j - i] = xp.dot(y_proj[i:j], x_proj[:src_size].T)
        simbwd[:j - i].max(axis=1, out=best_sim_backward[i:j])
        simbwd[:j - i] -= knn_sim_fwd / 2  # Equivalent to the real CSLS scores for NN
    if direction == 'forward':
        src_indices = src_indices_forward
        trg_indices = trg_indices_forward
    elif direction == 'backward':
        src_indices = src_indices_backward
        trg_indices = trg_indices_backward
    elif direction == 'union':
        src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
        trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))

    u, s, vt = xp.linalg.svd(y_proj[trg_indices].T.dot(x_proj[src_indices]))
    w1 = vt.T.dot(u.T)

    return w1