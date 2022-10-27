import numpy as np
import torch
import networkx as nx


class Environment(object):
    def __init__(self, graphs, args):
        self.graphs = graphs  # Graph Sequence
        self.seq_len = len(graphs)
        self.G = graphs[0]  # Current graph
        self.source = None
        self.current = None
        self.target = None
        self.observation = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.interval = args.interval

    def reset(self, subgraph, source, target, init_time):
        # todo: source, target, adj更新
        self.subgraph = subgraph
        self.subgraph_dict = dict(zip(range(len(subgraph)), subgraph))
        self.source = np.where(np.array(subgraph) == source)[0][0]
        self.target = np.where(np.array(subgraph) == target)[0][0]
        self.current = self.source
        self.observation = torch.Tensor([self.source, self.target]).type(torch.long)
        self.observation = self.observation.view(1, 2, 1).to(self.device)
        
        self.init_time = init_time
        self.current_time = init_time
        current_time_idx = self.get_current_time_idx(self.current_time)
        self.G = self.graphs[current_time_idx].subgraph(subgraph).copy()  # Current Graph
        self.time_list = [self.current_time]

    def get_current_time_idx(self, current_time):
        current_time_idx = int(current_time / self.interval) % self.seq_len
        return current_time_idx
    
    def observe(self):
        return self.observation.clone(), self.G.copy()

    def act(self, action) -> bool:
        done = True if action[-1] == self.target else False
        current_node = self.current
        for node in action:
            travel_time = self.G.get_edge_data(self.subgraph_dict[current_node], self.subgraph_dict[node])['travel_time']
            self.current_time += travel_time
            current_time_idx = self.get_current_time_idx(self.current_time)
            self.G = self.graphs[current_time_idx].subgraph(self.subgraph).copy()
            current_node = node

        self.time_list.append(self.current_time)
        self.current = action[-1]
        self.observation[:, 0, 0] = torch.Tensor([action[-1]]).to(self.device)
        return done

    def back(self, node: int) -> bool:
        done = False
        self.observation[:, 0, 0] = torch.Tensor([node]).to(self.device)
        
        # Update the current time and graph
        self.time_list.pop(-1)
        self.current_time = self.time_list[-1]
        current_time_idx = self.get_current_time_idx(self.current_time)
        self.G = self.graphs[current_time_idx].subgraph(self.subgraph).copy()
        self.current = node[0]

        return done

    def get_eval(self, actions: list):
        current = actions[0]
        current_time = self.init_time
        total_length = 0.0

        for i in range(1, len(actions)):
            target = actions[i]
            current_time_idx = self.get_current_time_idx(current_time)
            graph = self.graphs[current_time_idx]
            if not graph.get_edge_data(self.subgraph_dict[current], self.subgraph_dict[target]):
                print(current, target)
            current_time += graph.get_edge_data(self.subgraph_dict[current],
                                                self.subgraph_dict[target])['travel_time']
            total_length += graph.get_edge_data(self.subgraph_dict[current],
                                                self.subgraph_dict[target])['length']
            current = target

        travel_time = current_time - self.init_time
        return travel_time, total_length
