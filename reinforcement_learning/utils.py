import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple

from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import deque

from options import Options
opt = Options().parse()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps_done = 0

class DqnSolver:
    def __init__(self, env, policy_net, target_net, optimizer, memory = 10000):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.opt = optimizer
        self.memory = deque(maxlen=memory)
        self.criterion = nn.MSELoss()
        self.exploration_rate = 1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def change_optim(self, optimizer):
        self.opt = optimizer

    def act(self, s):
        if np.random.rand() < self.exploration_rate:
            return self.env.action_space.sample()
        else:
            s_tensor = torch.from_numpy(s).detach()
            Q_est = self.policy_net(s_tensor.float()).detach().numpy()
            # print("--------")
            # # print(np.argmax(Q_est))
            # print(Q_est)
            # print("--------")
            return np.argmax(Q_est)

    def experience_replay(self, loss_plt, batch_size = 64, gamma = 0.95, exploration_decay = 0.995, exploration_min = 0.01):
        global steps_done
        steps_done+=1
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in batch if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in batch if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in batch if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in batch if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e[4] for e in batch if e is not None]).astype(np.uint8)).float().to(
            device)

        # print(actions)

        Q_max_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        Q_target = rewards + gamma * Q_max_next * (1-dones)
        # print(Q_target)
        Q_estimated = self.policy_net(states).gather(1,actions)
        # print(Q_estimated)

        loss = self.criterion(Q_estimated, Q_target)
        self.opt.zero_grad()
        loss.backward()
        for name, param in self.policy_net.named_parameters():
            if 'mask' in name:
                continue
            param.grad.data.clamp_(-100, 100)
        self.opt.step()
        loss_plt.append(loss.detach().numpy().mean())
        # print(len(losss))
        self.exploration_rate *= exploration_decay
        self.exploration_rate = max(exploration_min, self.exploration_rate)

        if steps_done%opt.target_update == 0:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(policy_param.data)
