import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np


class GCN_QN3(torch.nn.Module):
    def __init__(self, input_features, reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T):
        super(GCN_QN3, self).__init__()
        # T: 计算graph embedding的迭代次数
        self.T = T
        self.embed_dim = embed_dim
        self.input_features = input_features
        # reg_hidden: 论文Eq.4 中， 对relu(...)输出的tensor做线性变换+relu(加层), 然后再将输入乘以theta5，得到一个数值, 即值函数的值
        self.reg_hidden = reg_hidden
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        #self.mu_1 = torch.nn.Linear(1, embed_dim)
        #torch.nn.init.normal_(self.mu_1.weight,mean=0,std=0.01)
        self.mu_1 = torch.nn.Parameter(torch.Tensor(self.input_features, embed_dim))
        torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim, True)
        torch.nn.init.normal_(self.mu_2.weight, mean=0, std=0.01)

        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            pre_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
            self.list_pre_pooling.append(pre_lin)

        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            post_lin =torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(post_lin.weight, mean=0, std=0.01)
            self.list_post_pooling.append(post_lin)

        self.q_1 = torch.nn.Linear(2*embed_dim, embed_dim, bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)
        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(2 * embed_dim, self.reg_hidden)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.q = torch.nn.Linear(self.reg_hidden, 1)
        else:
            self.q = torch.nn.Linear(2 * embed_dim, 1)
        torch.nn.init.normal_(self.q.weight, mean=0, std=0.01)

    def forward(self, xv, adj, input, actions):
        """
        :param xv: [batch_size, num_nodes, 1], partial solution set ==> 改为[batch_size, 2, 1]
        :param adj: [batch_size, num_nodes, num_nodes], 0-1邻接矩阵, Tensor
        :param input: [batch_size, num_nodes, 2], 点的二维坐标
        :param actions: [batch_size, num_action, 1]
        :return:
        """

        # batch的大小
        batch_size = xv.shape[0]
        num_node = input.shape[1]
        num_action = actions.shape[1]
        gv = adj

        for t in range(self.T):
            if t == 0:
                # 若t==0, 则
                #mu = self.mu_1(xv).clamp(0)
                # [batch_size, num_nodes, 1] x [1, embed_size] -> [batch_size, num_nodes, embed_size],
                # clamp: 令tensor中小于0的值->0，作用类似于relu
                # mu = torch.matmul(xv, self.mu_1).clamp(0)
                mu = torch.matmul(input, self.mu_1)
                mu = F.relu(mu)

                #mu.transpose_(1,2)
                #mu_2 = self.mu_2(torch.matmul(adj, mu_init))
                #mu = torch.add(mu_1, mu_2).clamp(0)
            else:
                #mu_1 = self.mu_1(xv).clamp(0)
                # [batch_size, num_nodes, 1] x [1, embed_size] -> [batch_size, num_nodes, embed_size]
                mu_1 = torch.matmul(input, self.mu_1).clamp(0)

                #mu_1.transpose_(1,2)
                # before pooling:
                for i in range(self.len_pre_pooling):
                    # mu: [batch_size, num_nodes, embed_size] -> [batch_size, num_nodes, embed_size] 对dim=2做线性变换
                    mu = self.list_pre_pooling[i](mu).clamp(0)

                # [num_nodes, num_nodes] x [batch_size, num_nodes, embed_size] -> [batch_size, num_nodes, embed_size]
                # 获得了聚合邻居节点embedding的新的embeddding, 若adj为0-1矩阵，则新的embedding为邻居节点embedding的加和
                #mu_pool = torch.matmul(adj, mu)
                mu_pool = torch.matmul(gv, mu)

                # after pooling
                for i in range(self.len_post_pooling):
                    # [batch_size, num_nodes, embed_size] -> [batch_size, num_nodes, embed_size]
                    # pooling后再次对embedding做多次线性变换+relu
                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)

                # [batch_size, num_nodes, embed_size] -> [batch_size, num_nodes, embed_size]
                mu_2 = self.mu_2(mu_pool)
                mu = torch.add(mu_1, mu_2).clamp(0)

        # source_v, target_v: [batch_size, 1, embed_size]
        source_v = mu[range(batch_size), xv[:, 0, 0], :].unsqueeze(1)
        target_v = mu[range(batch_size), xv[:, 1, 0], :].unsqueeze(1)
        # env_v: [batch_size, 1, embed_size * 2]
        env_v = torch.cat((source_v, target_v), 2)
        # [batch_size, 1, embed_size*2] ==> [batch_size, num_node, embed_size]
        q_1 = self.q_1(env_v).expand(batch_size, num_action, self.embed_dim)
        # 计算: theta7 * mu
        # [batch_size, num_action, embed_size]
        action_idx = actions.expand(batch_size, num_action, self.embed_dim)
        actions = torch.gather(input=mu, index=action_idx, dim=1)
        q_2 = self.q_2(actions)
        # 连接q_1和q_2，大小为[batch_size, num_nodes, embed_size*2]
        q_ = torch.cat((q_1, q_2), dim=-1)
        if self.reg_hidden > 0:
            # [batch_size, num_nodes, embed_size*2] -> [batch_size, num_nodes, reg_hidden]
            # relu(...)
            q_reg = self.q_reg(q_).clamp(0)
            # [batch_size, num_nodes, reg_hidden] -> [batch_size, num_nodes, 1]
            q = self.q(q_reg)
        else:
            q_ = q_.clamp(0)
            q = self.q(q_)
        # q: [batch_size, num_node, 1]
        return q


