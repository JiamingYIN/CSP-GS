class Runner_2hop(object):
    def __init__(self, environment, agent, args):
        self.environment = environment
        self.agent = agent
        self.ngames = args.ngames
        self.game_epoch = args.game_epoch
        self.max_iter = args.niter

    def reset(self, subgraph, source: int, target: int, init_time: float):
        self.environment.reset(subgraph, source, target, init_time)
        self.agent.reset(subgraph, source, target)

    def get_subpath(self, subgraph, source, target, init_time):
        self.reset(subgraph, source, target, init_time)
        subpath, travel_time, length = self.loop()
        if travel_time is None:
            return None, None, None
        map_dict = dict(zip(range(len(subgraph)), subgraph))
        subpath = [map_dict[v] for v in subpath]
        return subpath, travel_time, length

    def step(self, last_action=None):
        observation, current_graph = self.environment.observe()
        action, back = self.agent.act(observation, last_action, current_graph)

        if action is None:
            return (None, None, None)
        if not back:
            done = self.environment.act(action)
        else:
            done = self.environment.back(action)

        return (action, done, back)

    def loop(self):
        last_action = [self.environment.source]

        for _ in range(1, self.max_iter + 1):
            (act, done, back) = self.step(last_action=last_action)
            if act is None:
                return None, None, None
            if not back:
                # print('Iteration', _, ':', act)
                for v in act:
                    last_action.append(v)
            else:
                # print('Iteration', _, ':', 'Back to: ', act)
                del last_action[-1]
                del last_action[-1]
            if done:
                # print("Done at step ", _, "  Number of nodes: ", len(last_action))
                break

        travel_time, length = self.environment.get_eval(last_action)
        return last_action, travel_time, length
