import networkx as nx
import numpy as np
import pandas as pd
import os
import folium
import torch
from scipy.sparse import csr_matrix
import scipy.sparse as sp


def remove_bd_edge(G, threshold=500):
    removed_edges = []
    for e in G.edges:
        if G[e[0]][e[1]]['travel_time'] >= threshold:
            removed_edges.append(e)
    Gnew = G.copy()
    for re in removed_edges:
        Gnew.remove_edge(re[0], re[1])
    return Gnew

def save_dict(d, path):
    if isinstance(d, str):
        d = eval(d)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(d))


def load_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = eval(f.read())
    return d

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_adj(G_nx, nodelist, model_size=2030):
    adj = nx.adjacency_matrix(G_nx, nodelist=nodelist, weight='travel_time')
    max_weight, min_weight = adj.max(), adj.min()
    adj = (adj - min_weight) / (max_weight - min_weight)
    adj_t = adj.transpose()

    # Padding
    node_num = len(nodelist)
    remain_ind = model_size - node_num
    bottom_mat = csr_matrix((remain_ind, remain_ind))
    adj = sp.block_diag((adj, bottom_mat))
    adj_t = sp.block_diag((adj_t, bottom_mat))

    # Covert to Tensor
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_t = sparse_mx_to_torch_sparse_tensor(adj_t)
    return adj, adj_t

def get_evaluation(graphs: list = None,
                   interval: int = 30,
                   path: list = None):
    """
    Calculate the real travel time in the dynamic graph for a given path.
    :param graphs: List of graphs
    :param interval: the time interval of the graph sequence
    :param path: one path, a list of node
    """
    current = path[0]
    length = 0
    travel_time = 0
    for p in path[1:]:
        current_time_idx = int(travel_time / interval)
        if current_time_idx >= len(graphs):
            current_time_idx = current_time_idx % len(graphs)
        G = graphs[current_time_idx]
        length += G[current][p]['length']
        travel_time += G[current][p]['travel_time']
        current = p
    return length, travel_time


def check_save_path(save_path):
    if not os.path.exists(save_path):
        logs = pd.DataFrame(columns=['method',
                                     'dir_path',
                                     'random_seed',
                                     'avg(Length / R)',
                                     'Average Length',
                                     'Average Travel Time',
                                     'avg(travel_time/optimal_travel_time)',
                                     'Feasible Ratio',
                                     'Time'])
        logs.to_csv(save_path)

def weight_matrix(adj):
    adj = adj / 1000.
    W2 = np.multiply(adj, adj)
    W2[W2 == 0] = np.inf
    sigma2 = 0.1
    epsilon = 0.1
    W = np.multiply(np.exp(-W2 / sigma2), (np.exp(-W2 / sigma2) >= epsilon))
    return W


def get_normalized_adj(A):
    D = A.copy()
    D[D > 0] = 1
    D = np.array(np.sum(D, axis=1)).reshape((-1,))
    # D[D <= 10e-5] = 10e-5  # Prevent infs
    D[D <= 10e-5] = 10e5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))

    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    # A_hat = np.multiply(diag.reshape((-1, 1)), A)
    A_hat = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                        diag.reshape((1, -1)))
    return A_hat


def get_random_walk_matrix(A):
    A_ = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    d_inv = np.reciprocal(D)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.diag(d_inv)
    A_hat = np.dot(d_mat_inv, A_)
    return A_hat


def get_graph_features(dist_matrix):
    dist_matrix = np.mat(dist_matrix)
    # normalize: 0-1
    dist_matrix = dist_matrix / dist_matrix.max()
    temp = dist_matrix.copy()
    temp[dist_matrix == 0] = -np.inf
    # Construct Node Features: [max_dist, min_dist, avg_dist, in_degree, out_degree]
    max_dist_in = temp.max(axis=0).reshape(-1, 1)
    max_dist_out = temp.max(axis=1)

    temp[dist_matrix == 0] = np.inf
    min_dist_in = temp.min(axis=0).reshape(-1, 1)
    min_dist_out = temp.min(axis=1)

    degree_matrix = dist_matrix.copy()
    degree_matrix[degree_matrix > 0] = 1
    in_degree = degree_matrix.sum(axis=0).reshape(-1, 1)
    out_degree = degree_matrix.sum(axis=1)

    avg_dist_in = np.divide(dist_matrix.sum(axis=0).reshape(-1, 1), in_degree)
    avg_dist_out = np.divide(dist_matrix.sum(axis=1), out_degree)
    avg_dist_in[np.isinf(avg_dist_in)] = 0
    avg_dist_out[np.isinf(avg_dist_out)] = 0

    graph_features = np.concatenate([max_dist_in, min_dist_in, avg_dist_in, max_dist_out, min_dist_out, avg_dist_out,
                                     in_degree, out_degree], axis=1)
    return graph_features


def save_confs(args, save_path):
    conf_dict = args.__dict__
    with open(os.path.join(save_path, 'confs.txt'), 'w') as file:
        for each_arg, value in conf_dict.items():
            file.writelines(each_arg + ' : ' + str(value) + '\n')
