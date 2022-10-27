import networkx as nx
from collections import deque
from heapq import heappush, heappop
import numpy as np
from data.data_loader import DataLoader
import time

def get_predecessor(G, source):
    G_succ = G._succ if G.is_directed() else G._adj
    push = heappush
    pop = heappop
    dist = {}
    seen = {}
    fringe = []  # heapq with 3-tuples (distance, c, node)
    seen[source] = 0
    push(fringe, (0, source))
    pred = {}

    while fringe:
        (d, v) = pop(fringe)
        if v in dist:
            continue  # already searched this node
        dist[v] = d
        for u, e in G_succ[v].items():
            cost = e['length']
            vu_dist = dist[v] + cost
            if u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, u))
                pred[u] = v

    succ = {}
    for v, v_pred in pred.items():
        if v_pred not in succ:
            succ[v_pred] = [v]
        else:
            succ[v_pred].append(v)

    return pred, succ, seen

def build_path_from_predecessors(source, target, pred):
    v = target
    path = [int(v)]
    while v != source:
        if v in pred.keys():
            v = pred[v]
            path.append(int(v))
        else:
            return None
    return path

def get_paths(G, source, target, K=20, R=None):
    fwd_pred, fwd_succ, fwd_cost = get_predecessor(G, source)
    bwd_pred, bwd_succ, bwd_cost = get_predecessor(G.reverse(), target)

    joint_nodes = deque([('forward', source), ('backward', target)])

    cnt = 1

    sp = build_path_from_predecessors(source, target, fwd_pred)
    sp.reverse()
    paths = [sp]

    while cnt < K and len(joint_nodes) > 0:
        direc, joint_node = joint_nodes.popleft()
        if direc == 'forward':
            if joint_node in fwd_succ:
                extend_nodes = fwd_succ[joint_node]
                for n in extend_nodes:
                    joint_nodes.append(('forward', n))
        else:
            if joint_node in bwd_succ:
                extend_nodes = bwd_succ[joint_node]
                for n in extend_nodes:
                    joint_nodes.append(('backward', n))
        fwd_path = build_path_from_predecessors(source, joint_node, fwd_pred)
        bwd_path = build_path_from_predecessors(target, joint_node, bwd_pred)
        if fwd_path is None or bwd_path is None:
            continue

        if len(set(fwd_path[1:]).intersection(set(bwd_path))) == 0:
            if R is None or fwd_cost[joint_node] + bwd_cost[joint_node] <= R:
                fwd_path.reverse()
                path = fwd_path[:-1] + bwd_path
                paths.append(path)
                cnt += 1

    return paths

if __name__ == '__main__':
    data = DataLoader('data/map-2030-5-5-5-w-75-p-175-420-uniform')
    G = data.get_graph()
    sps = data.get_sp_path()
    t1 = time.time()
    for i in range(len(data.sources)):
        sp = sps[i]
        source = data.sources[i]
        target = data.targets[i]
        paths = get_paths(G, source, target, K=1)
    t2 = time.time()

    print(t2 - t1)