import os
import utils
import networkx as nx
import torch
import numpy as np
from .models import GCN_QN3

from utils import weight_matrix, get_normalized_adj, get_graph_features, get_random_walk_matrix, remove_bd_edge

def get_2hop_neighbors(G, node_dict):
    N = G.number_of_nodes()
    new_adj = np.zeros(shape=(N, N))
    for v in G.nodes():
        ns1 = set()
        ns2 = set()
        for u in G.neighbors(v):
            if u != v:
                ns1.add(node_dict[u])
                for w in G.neighbors(u):
                    if w != u and w != v:
                        ns2.add(node_dict[w])
        actions = ns2 - ns1
        actions = list(actions)
        new_adj[node_dict[v]][actions] = 1
    return new_adj

def get_valid_actions(adj, adj_2hop, target):
    direct_nodes = np.argwhere(adj[:, target] > 0)
    adj_2hop[direct_nodes, target] = 1
    return adj_2hop

class Agent_2hop:
    def __init__(self, graph, args):
        # Graph Embedding相关
        self.graph = graph
        self.embed_dim = args.embed_dim
        self.model_type = args.model
        self.num_nodes = nx.number_of_nodes(self.graph.G)
        self.input_features = args.input_features
        self.t = args.t
        # cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model
        self.Q = GCN_QN3(input_features=args.input_features, reg_hidden=args.embed_dim, embed_dim=args.embed_dim,
                                    len_pre_pooling=0, len_post_pooling=0, T=self.t)

        if args.pretrain is not None:
            if args.cuda is None:
                pretrained_dict = torch.load(os.path.join(os.getcwd(), args.pretrain), map_location=torch.device('cpu'))
            else:
                pretrained_dict = torch.load(os.path.join(os.getcwd(), args.pretrain))
            self.Q.load_state_dict(pretrained_dict)
            # print("Successfully load model!")

        if args.cuda is not None:
            self.Q = self.Q.cuda(args.cuda)

    def reset(self, subgraph: list, source: int, target: int):
        self.subgraph = subgraph
        self.node_dict = dict(zip(subgraph, range(len(subgraph))))
        self.inv_node_dict = dict(zip(range(len(subgraph)), subgraph))
        self.num_nodes = len(subgraph)
        self.source = self.node_dict[source]
        self.target = self.node_dict[target]

        node_input = self.graph.nodes.copy()[subgraph, :]
        node_input[:, 0] = (node_input[:, 0] - min(node_input[:, 0])) / (max(node_input[:, 0]) - min(node_input[:, 0]))
        node_input[:, 1] = (node_input[:, 1] - min(node_input[:, 1])) / (max(node_input[:, 1]) - min(node_input[:, 1]))

        self.rec_covered = np.zeros((self.num_nodes))
        self.rec_covered[self.source] = 1
        self.target_flag = np.zeros((self.num_nodes))
        self.target_flag[self.target] = 1

        subG = self.graph.G.subgraph(subgraph).copy()  # 利用nx.Digraph.subgraph, 根据node_list获得G的子图
        subG = utils.remove_bd_edge(subG)

        raw_adj = nx.adjacency_matrix(subG, nodelist=subgraph)
        self.raw_adj = np.array(raw_adj.todense())  # Original 0/1 matrix

        tt_matrix = nx.adjacency_matrix(subG, nodelist=subgraph, weight='travel_time')
        tt_matrix = np.array(tt_matrix.todense())

        resource_matrix = nx.adjacency_matrix(subG, nodelist=subgraph, weight='length').todense()
        resource_features = get_graph_features(resource_matrix)
        graph_features = get_graph_features(tt_matrix.copy())
        self.node_input_arr = np.concatenate([node_input, self.target_flag.reshape(-1, 1), resource_features, graph_features], axis=1)

        # Weighted adjacency matrix
        A = weight_matrix(tt_matrix)
        A_hat = get_random_walk_matrix(A)
        self.adj = torch.from_numpy(np.expand_dims(A_hat.astype(float), axis=0))
        self.adj = self.adj.type(torch.FloatTensor).to(self.device)

        # self.adj_2hop = get_2hop_neighbors(subG)
        # self.valid_actions = get_valid_actions(self.raw_adj, self.adj_2hop.copy(), self.target)

    def act(self, observation: torch.Tensor, last_action: list = None, current_graph: nx.DiGraph = None):
        """
        :param observation: [1, num_nodes, 1], 元素为0或1, 0代表未被选择过
        :param last_action: list, current path
        :param current_graph
        :return:
        """
        back = False
        obs = observation.cpu().numpy()
        current = obs[0, 0, 0]
        current_graph = remove_bd_edge(current_graph)
        adj = nx.adjacency_matrix(current_graph, nodelist=self.subgraph)
        adj = np.array(adj.todense())
        adj_2hop = get_2hop_neighbors(current_graph, self.node_dict)
        valid_actions = get_valid_actions(adj, adj_2hop.copy(), self.target)
        actions = np.array(list(range(self.num_nodes)))

        nbrs = np.where(valid_actions[current, :] > 0)[0].tolist()
        mask = np.isin(actions, nbrs)
        uncovered_nodes = np.argwhere(self.rec_covered == 0)
        uncovered_mask = np.isin(actions, uncovered_nodes)
        mask = mask * uncovered_mask
        candidates = actions[mask]

        # Update the node input
        tt_adj_matrix = nx.adjacency_matrix(current_graph, nodelist=self.subgraph, weight='travel_time')
        tt_adj_matrix = np.array(tt_adj_matrix.todense())
        tt_features = get_graph_features(tt_adj_matrix)
        self.node_input_arr[:, -8:] = tt_features
        node_input_tensor = torch.from_numpy(self.node_input_arr).float().unsqueeze(dim=0).to(self.device)

        if len(candidates) == 0:
            # 需要回退
            if len(last_action) < 3:
                action = None
            else:
                action = [last_action[-3]]
            back = True

        else:
            candidate_actions = np.array(candidates)
            candidate_actions = torch.from_numpy(candidate_actions).to(self.device)
            candidate_actions = candidate_actions.view(1, len(candidates), 1)
            q_a = self.Q(observation, self.adj, node_input_tensor, candidate_actions)
            q_a = q_a.cpu().detach().numpy()
            q_a = q_a[0, :, 0]
            action = np.argmax(q_a)
            action = candidates[action]

        if action is not None and not back:
            self.rec_covered[action] = 1

            nodes = []
            # if not (action == self.target and self.raw_adj[current][self.target] == 1):
            preds = list(current_graph.predecessors(self.inv_node_dict[action]))
            succs = list(current_graph.successors(self.inv_node_dict[current]))
            optimal_nodes, opt_w = None, 100000.0
            for u in succs:
                if u == self.inv_node_dict[action]:
                    opt_w = current_graph.get_edge_data(self.inv_node_dict[current], u)['travel_time']
                    optimal_nodes = [action]
                    break
                for v in preds:
                    if u == v:
                        weight = current_graph.get_edge_data(self.inv_node_dict[current], u)['travel_time'] + \
                                 current_graph.get_edge_data(u, self.inv_node_dict[action])['travel_time']
                        nodes = [self.node_dict[u], action]
                        if weight < opt_w:
                            opt_w = weight
                            optimal_nodes = nodes

                # for n in current_graph.successors(self.inv_node_dict[current]):
                #     if n in preds:
                #         weight = current_graph.get_edge_data(self.inv_node_dict[current], n)['travel_time'] + \
                #                  current_graph.get_edge_data(n, self.inv_node_dict[action])['travel_time']
                #         if weight < opt_w:
                #             opt_w = weight
                #             optimal_v1 = n
                # nodes.append(self.node_dict[optimal_v1])
            # nodes.append(action)
            # action = nodes

            action = optimal_nodes

        return action, back
