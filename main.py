from csp_dqn_2hop import Runner_2hop, Agent_2hop, Environment
import datetime
import numpy as np
from csp_gs import csp_gs
from data.data_loader import DataLoader
import torch
import argparse
import pandas as pd
import os
import random
import warnings
from utils import save_confs

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()

parser.add_argument('--dir_path', type=str, default='koln_data/koln1630_1.5',
                    help='path of the data files: data/map-2030-50-50-1-150-500-new, koln_data/koln1630_1.5')
parser.add_argument('--method', type=str, default='CSP-GS')
parser.add_argument('--ngames', type=int, default=100, help='number of games to simulate')
parser.add_argument('--niter', type=int, default=1000, help='max number of iterations per game')
parser.add_argument('--train_steps', type=int, default=50, help='number of train steps for each game')
parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
parser.add_argument('--game_epoch', type=int, default=1)
parser.add_argument('--bs', type=int, default=64, help="minibatch experience size for training")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--model', type=str, default='GCN-QN2', help='model name')
parser.add_argument('--input_features', type=int, default=19)
parser.add_argument('--n_step', type=int, default=15, help="n steps in RL")
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--t', type=int, default=3)
parser.add_argument('--cuda', default=None)
parser.add_argument('--pretrain', default='models/model2nd.pt', help='pretrained model')
parser.add_argument('--min_score', type=float, default=0.05, help='Minimum score for sampling')
parser.add_argument('--sample_time', default=150, type=int, help='Sampling time')
parser.add_argument('--random_seed', type=int, default=1000)
parser.add_argument('--lambda_rc', default=0.5, type=float)
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('--log_path', type=str, default='test.csv')
parser.add_argument('--n', type=int, default=5)

def main():
    args = parser.parse_args()
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if args.cuda is not None:
        torch.cuda.set_device(args.cuda)

    data = DataLoader(dir_path=args.dir_path)
    sources = data.sources
    targets = data.targets
    Rs = data.Rs
    graphs = data.graph_seq
    opt_tts = data.opt_travel_time

    env = Environment(data.graph_seq, args)

    # Save Results
    logs = pd.DataFrame(columns=['game', 'path', 'travel_time', 'opt_travel_time', 'ratio', 'length', 'R',
                                 'constraint_ratio', 'feasibility', 'running_time'])
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y%m%d%H%M')
    save_path = os.path.join('results', time_str)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_confs(args, save_path)
    logs.to_csv(os.path.join(save_path, 'log.csv'))

    logs = []

    time_statistic = {'DQN-Time': 0.0,
                      'Sampler-Time': 0.0,
                      'Select-Time': 0.0,
                      'Process-Time': 0.0}

    for i in range(len(sources)):
        source = sources[i]
        target = targets[i]
        R = Rs[i]
        nodes = list(range(data.V))

        agent = Agent_2hop(data, args)
        dqn_runner = Runner_2hop(env, agent, args)
        path, length, travel_time, time_dict = csp_gs(dqn_runner, graphs, data.nodes, nodes, source,
                                                      target, R, 0, args)
        

        running_time =  time_dict['DQN-Time'] + time_dict['Sampler-Time'] + time_dict['Select-Time']
        time_statistic['DQN-Time'] += time_dict['DQN-Time']
        time_statistic['Sampler-Time'] += time_dict['Sampler-Time']
        time_statistic['Select-Time'] += time_dict['Select-Time']

        opt_tt = opt_tts[i]
        feasibility = True if length <= R + 0.0001 else False
        ratio = travel_time / opt_tt
        cratio = length / R
        one_log = [[i, path, travel_time, opt_tt, ratio, length, R, cratio, feasibility, running_time]]
        pd.DataFrame(one_log).to_csv(os.path.join(save_path, 'log.csv'), header=None, mode='a')
        logs.append(one_log[0])


    logs = pd.DataFrame(logs, columns=['game', 'path', 'travel_time', 'opt_travel_time', 'ratio', 'length', 'R',
                                       'constraint_ratio', 'feasibility', 'running_time'])

    exp_record = {}
    exp_record['avg(Length / R)'] = logs['constraint_ratio'].mean()
    exp_record['Average Length'] = logs['length'].mean()
    exp_record['Average Travel Time'] = logs['travel_time'].mean()
    exp_record['avg(travel_time/optimal_travel_time)'] = logs['ratio'].mean()
    exp_record['Feasible Ratio'] = logs[logs['feasibility'] == True].shape[0] / logs.shape[0]
    exp_record['Time'] = logs['running_time'].sum()

    results = [args.method, args.dir_path, args.random_seed]

    with open(os.path.join(save_path, 'exp_record.txt'), 'w') as file:
        for each_arg, value in exp_record.items():
            file.writelines(each_arg + ' : ' + str(value) + '\n')
            print(each_arg + ' : ' + str(value))
            results.append(value)

    if not os.path.exists(args.log_path):
        logs = pd.DataFrame(columns=['method',
                                     'dir_path',
                                     'random_seed',
                                     'avg(Length / R)',
                                     'Average Length',
                                     'Average Travel Time',
                                     'avg(travel_time/optimal_travel_time)',
                                     'Feasible Ratio',
                                     'Time'])
        logs.to_csv(args.log_path)

    pd.DataFrame([results]).to_csv(args.log_path, mode='a', header=None)

if __name__ == '__main__':
    main()
