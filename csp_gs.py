import numpy as np
import random
import math
import time
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path
from sp_utils import get_paths


class CSPSampler_static:
    def __init__(self, graphs, node_locs, args):
        self.m = args.sample_time
        self.node_locs = node_locs
        self.min_score = args.min_score
        self.lambda_rc = args.lambda_rc

        self.graphs = graphs
        self.seq_len = len(graphs)
        self.max_iter = 100000
        self.interval = args.interval
        self.V = self.node_locs.shape[0]

        self.one_graph = self.graphs[0]

    def update(self, nodes, source, target, R, init_time):
        self.nodes = nodes
        self.one_graph = self.graphs[0].subgraph(nodes)
        self.source = source
        self.target = target
        self.init_time = init_time
        self.V = len(nodes)
        self.R = R
        self.nodes = nodes

    def extend_graph(self):
        G = self.graphs[0]
        nodes = set(self.nodes)
        for v in self.nodes:
            for n in G.neighbors(v):
                nodes.add(n)

        self.nodes = nodes
        self.one_graph = self.graphs[0].subgraph(nodes)
        self.V = len(nodes)

    def get_eval(self, path, init_time):
        current = path[0]
        dist = 0
        current_time = init_time

        for p in path[1:]:
            # Latest id of current time unit
            current_time_idx = int(current_time / self.interval)
            if current_time_idx >= self.seq_len:
                current_time_idx = current_time_idx % self.seq_len
            # Length and Travel time of the new edge
            dist += self.graphs[current_time_idx][current][p]['length']
            current_time += self.graphs[current_time_idx][current][p]['travel_time']
            current = p

        travel_time = current_time - init_time

        return dist, travel_time

    def get_bidirec_sps(self, source, target):
        fwd_paths = single_source_dijkstra_path(self.one_graph, source)
        bwd_paths = single_source_dijkstra_path(self.one_graph, target)
        return fwd_paths, bwd_paths

    def check_reachability(self, source, target):
        has_path = False
        while not has_path:
            path = self.sample_one_path(source, target)
            if path is None:
                self.extend_graph()
            else:
                has_path = True

    def sample_m_path(self, source, target):
        paths = get_paths(self.one_graph, source, target, self.m, None)
        dists = []
        travel_times = []
        for i in range(len(paths)):
            path = paths[i]
            dist, travel_time = self.get_eval_static(path, self.init_time)
            dists.append(dist)
            travel_times.append(travel_time)

        return paths, dists, travel_times

    def sample_one_path(self, source, target):
        state = dict(zip(list(self.one_graph.nodes), [0] * self.V))
        current = source
        state[source] = 1
        path = [source]

        for i in range(self.max_iter):
            neighbors = [int(n) for n in self.one_graph.neighbors(current)]
            candidates = []
            for neighbor in neighbors:
                if state[neighbor] == 0:
                    candidates.append(neighbor)
            if len(candidates) > 0:
                action = random.choice(candidates)
                path.append(action)
                state[action] = 1
                current = action
            else:
                if len(path) == 1:
                    break
                current = path[-2]
                path.pop(-1)

            if current == target:
                break

        if len(path) == 1:
            print("Invalid Graph!")
            return None, None, None

        return path

    def get_rcs(self, path) -> np.array:
        rcs = [0]
        rcp = 0
        current = path[0]
        for p in path[1:]:
            rcp += self.one_graph[current][p]['length']
            rcs.append(rcp)
            current = p
        return np.array(rcs)

    def get_dist(self, path):
        current = path[0]
        dist = 0
        for p in path[1:]:
            dist += self.one_graph[current][p]['length']
            current = p
        return dist

    def get_eval_static(self, path, init_time):
        # Evaluate the travel time of the path based on current graph
        # Return: Distance, Travel Time

        current = path[0]
        dist = 0.0
        travel_time = 0.0
        current_time = init_time
        current_time_idx = int(current_time / self.interval) % len(self.graphs)
        # current_time_idx = 0
        for p in path[1:]:
            # Length and Travel time of the new edge
            dist += self.graphs[current_time_idx][current][p]['length']
            travel_time += self.graphs[current_time_idx][current][p]['travel_time']
            current = p

        return dist, travel_time

    def select_nodes(self, score_dict):
        scores = np.array(list(score_dict.values()))
        min_score = self.min_score
        if sum(scores > min_score) < 10 or sum(scores > min_score) == self.V:
            min_score = np.percentile(scores, 30)

        while sum(scores > min_score) == self.V:
            min_score = min_score + 0.01

        sampled_nodes = []
        for v in self.one_graph.nodes():
            if score_dict[v] > min_score:
                sampled_nodes.append(v)

        return sampled_nodes

    def get_path_score(self, travel_time, length):
        if self.tb_cal[0] == 0:
            tt_bonus = 1.0
        else:
            tt_bonus = math.exp(-(travel_time - self.min_tt) ** 2 / (self.tb_cal[0] ** 2))
        tt_bonus = tt_bonus / self.tb_cal[1]

        if self.lb_cal[0] == 0:
            l_bonus = 1.0
        else:
            l_bonus = math.exp(-max(0, length - self.R) ** 2 / (self.lb_cal[0] ** 2))
        l_bonus = l_bonus / self.lb_cal[1]

        score = self.lambda_rc * l_bonus + (1 - self.lambda_rc) * tt_bonus
        return score

    def sample(self, nodes, source, target, R, init_time):
        self.update(nodes, source, target, R, init_time)
        paths, lengths, tts = self.sample_m_path(source, target)
        m = len(paths)

        score = dict(zip(list(self.one_graph.nodes), [0] * self.V))
        cnt = dict(zip(list(self.one_graph.nodes), [0] * self.V))
        sigma = np.std(tts)
        min_tt = min(tts)

        if sigma != 0:
            tt_bonus = [math.exp(-(t - min_tt) ** 2 / (sigma ** 2)) for t in tts]
        else:
            tt_bonus = [1.0 for _ in tts]
        sum_tt_bonus = sum(tt_bonus)
        tt_bonus = [b / sum_tt_bonus for b in tt_bonus]

        # 根据每条路径的Length, 计算每条路径的length bonus
        # Calculate the length bonus for each path
        sigma_l = np.std(lengths)
        if sigma_l > 0.01:
            length_bonus = [math.exp(-max(0, l - self.R) ** 2 / (sigma_l ** 2)) for l in lengths]
        else:
            length_bonus = [1.0 for _ in lengths]
        sum_length_bonus = sum(length_bonus) + 0.001

        length_bonus = [lb / sum_length_bonus for lb in length_bonus]

        opt_path_ind = 0
        min_tt = tts[opt_path_ind]

        for i in range(m):
            if tts[i] < min_tt and lengths[i] < self.R + 0.001:
                min_tt = tts[i]
                opt_path_ind = i
            for p in paths[i]:
                score[p] += length_bonus[i] * self.lambda_rc + tt_bonus[i] * (1 - self.lambda_rc)
                cnt[p] += 1

        # Save for the path score comparing
        self.min_tt = min_tt
        self.best_pi = length_bonus[opt_path_ind] * self.lambda_rc + tt_bonus[opt_path_ind] * (1 - self.lambda_rc)
        self.lb_cal = [sigma_l, sum_length_bonus]
        self.tb_cal = [sigma, sum_tt_bonus]

        crucial_score = {}
        for v in list(self.one_graph.nodes):
            crucial_score[v] = score[v]

        opt_path = paths[opt_path_ind]

        # Calculate the real weight in dynamic graph, only used for updating the time step
        _, opt_tt = self.get_eval(opt_path, self.init_time)
        opt_length = lengths[opt_path_ind]
        self.opt_path = opt_path

        sampled_nodes = self.select_nodes(score)
        sampled_nodes = list(set(sampled_nodes + opt_path))
        sampled_nodes = sorted(sampled_nodes)

        node_rc = self.get_rcs(opt_path)

        return sampled_nodes, opt_path, opt_tt, opt_length, node_rc


def csp_gs(runner, graphs, node_locs, nodes, source, target, R, init_time, args):
    n = args.n
    sampler = CSPSampler_static(graphs, node_locs, args)

    current_time = init_time
    current_res_consumption = 0.
    current_node = source
    path = [source]

    dqn_time = 0.
    sampler_time = 0.
    select_time = 0.

    t1 = time.time()
    subgraph, opt_path, opt_tt, opt_length, rcs = sampler.sample(nodes, source, target,
                                                                         R - current_res_consumption, current_time)
    t2 = time.time()
    sampler_time += t2 - t1

    while True:
        if len(opt_path) > n:
            t1 = time.time()
            vstar_ind = n - 1
            vstar = opt_path[vstar_ind]
            t2 = time.time()
            select_time += t2 - t1
        else:
            vstar = target

        # current node -> vstar
        t1 = time.time()
        subgraph1, sub_path, opt_tt, opt_length, rcs = sampler.sample(subgraph, current_node, vstar,
                                                                              R - current_res_consumption, current_time)
        t2 = time.time()
        sampler_time += t2 - t1

        t1 = time.time()
        dqn_path, dqn_tt, dqn_length = runner.get_subpath(subgraph1, current_node, vstar, current_time)
        t2 = time.time()
        dqn_time += t2 - t1

        if dqn_path is not None:
            _, static_dqn_tt = sampler.get_eval_static(dqn_path, current_time)
            _, static_pstar_tt = sampler.get_eval_static(sub_path, current_time)
            s_pstar = sampler.min_score
            s_dqnpath = sampler.get_path_score(static_dqn_tt, dqn_length)

            if s_pstar <= s_dqnpath:
                path += dqn_path[1:]
                current_time += dqn_tt
                current_res_consumption += dqn_length
            else:
                path += sub_path[1:]
                current_time += opt_tt
                current_res_consumption += opt_length
        else:
            path += sub_path[1:]
            current_time += opt_tt
            current_res_consumption += opt_length

        # vstar -> target
        if vstar == target:
            break

        t1 = time.time()
        subgraph, opt_path, opt_tt, opt_length, rcs = sampler.sample(subgraph, vstar, target,
                                                                             R - current_res_consumption, current_time)
        t2 = time.time()
        sampler_time += t2 - t1
        current_node = vstar

    running_time = {'DQN-Time': dqn_time,
                    'Select-Time': select_time,
                    'Sampler-Time': sampler_time}

    dist, travel_time = sampler.get_eval(path, 0.0)
    return path, dist, travel_time, running_time
