###
'''
June 2019
Code by: Arnaud Fickinger
'''
###

import gym
import math
import random
import numpy as np
import matplotlib


matplotlib.use('agg')

import matplotlib.pyplot as plt
from collections import deque

from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils import *
from model import *

import cv2 as cv

from options import Options
opt = Options().parse()

env = gym.make('CartPole-v0')

space_dim = 4
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

else:
    torch.set_default_tensor_type(torch.FloatTensor)

loss_plt = []
reward_100 = []

def main():
    overall_acc_0_init = []
    overall_acc_4_init = []
    overall_acc_20_init = []
    overall_acc_60_init = []

    overall_acc_0_rand = []
    overall_acc_4_rand = []
    overall_acc_20_rand = []
    overall_acc_60_rand = []

    pruning = [4, 20, 60]
    # pruning = [4, 20, 60]

    lbls = ['0', '4', '20', '60']

    for test in range(10):

        print("test {}".format(test))

        policy_net = FCNet(space_dim, 300, 100, action_dim)
        target_net = FCNet(space_dim, 300, 100, action_dim)

        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(policy_param.data)

        optimizer = torch.optim.Adam([w for name, w in policy_net.named_parameters() if not 'mask' in name],
                                     lr=opt.lr)  # try with that

        Solver = DqnSolver(env, policy_net, target_net , optimizer)
        rewards = []
        mean_100_rewards_0 = []

        all_mean_100_rewards = []
        for episode in range(opt.episodes):
            # print(episode)
            R = 0
            s = env.reset()
            step = 0
            while (True):
                step += 1
                a = Solver.act(s)
                s_next, r, done, _ = env.step(a)
                env.render()
                R += r
                Solver.remember(s, a, r, s_next, done)
                if done:
                    # print(R)
                    rewards.append(R)
                    if (len(rewards) > 100):
                        mean_100_rewards_0.append(np.mean(rewards[-100:]))
                    else:
                        mean_100_rewards_0.append(np.mean(rewards))
                    break
                Solver.experience_replay(loss_plt)
                s = s_next

        all_mean_100_rewards.append(mean_100_rewards_0)

        policy_net.save_trained_weight()

        print("smart")

        for rate in pruning:

            print("rate: {}".format(rate))
            tmpr_mean_100_reward = []
            output_rate = int(rate/2)

            optimizer = torch.optim.Adam([w for name, w in policy_net.named_parameters() if not 'mask' in name], lr=opt.lr) #try with that
            Solver.change_optim(optimizer)

            policy_net.load_trained_weight()
            policy_net.reset_mask()
            policy_net.prune(rate, output_rate)
            policy_net.reinitializ()

            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(policy_param.data)

            rewards = []

            for episode in range(opt.episodes):
                # print(episode)
                R = 0
                s = env.reset()
                step = 0
                while (True):
                    step += 1
                    a = Solver.act(s)
                    s_next, r, done, _ = env.step(a)
                    R += r
                    Solver.remember(s, a, r, s_next, done)
                    if done:
                        # print(R)
                        rewards.append(R)
                        if (len(rewards) > 100):
                            tmpr_mean_100_reward.append(np.mean(rewards[-100:]))
                        else:
                            tmpr_mean_100_reward.append(np.mean(rewards))
                        break
                    Solver.experience_replay(loss_plt)
                    s = s_next

            all_mean_100_rewards.append(tmpr_mean_100_reward)

        plt.clf()
        for acc, lbl in zip(all_mean_100_rewards, lbls):
            plt.plot(np.arange(len(acc)), acc, label=lbl)
        plt.legend(title="Pruning (%):")
        plt.xlabel("Iteration")
        plt.ylabel("Mean of 100 last rewards")
        plt.savefig("lotteryticket_rl_smart_init_{}".format(test))
        plt.close()

        overall_acc_0_init.append(all_mean_100_rewards[0])
        overall_acc_4_init.append(all_mean_100_rewards[1])
        overall_acc_20_init.append(all_mean_100_rewards[2])
        overall_acc_60_init.append(all_mean_100_rewards[3])

        all_mean_100_rewards = []
        all_mean_100_rewards.append(mean_100_rewards_0)

        print("rand")

        for rate in pruning:

            print("rate: {}".format(rate))
            tmpr_mean_100_reward = []
            output_rate = int(rate / 2)

            optimizer = torch.optim.Adam([w for name, w in policy_net.named_parameters() if not 'mask' in name],
                                         lr=opt.lr)  # try with that

            Solver.change_optim(optimizer)

            policy_net.load_trained_weight()
            policy_net.reset_mask()
            policy_net.prune(rate, output_rate)
            policy_net.random_reinit()

            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(policy_param.data)

            rewards = []

            for episode in range(opt.episodes):
                # print(episode)
                R = 0
                s = env.reset()
                step = 0
                while (True):
                    step += 1
                    a = Solver.act(s)
                    s_next, r, done, _ = env.step(a)
                    R += r
                    Solver.remember(s, a, r, s_next, done)
                    if done:
                        # print(R)
                        rewards.append(R)
                        if (len(rewards) > 100):
                            tmpr_mean_100_reward.append(np.mean(rewards[-100:]))
                        else:
                            tmpr_mean_100_reward.append(np.mean(rewards))
                        break
                    Solver.experience_replay(loss_plt)
                    s = s_next

            all_mean_100_rewards.append(tmpr_mean_100_reward)

        plt.clf()
        for acc, lbl in zip(all_mean_100_rewards, lbls):
            plt.plot(np.arange(len(acc)), acc, label=lbl)
        plt.legend(title="Pruning (%):")
        plt.xlabel("Iteration")
        plt.ylabel("Mean of 100 last rewards")
        plt.savefig("lotteryticket_rl_rand_init_{}".format(test))
        plt.close()

        overall_acc_0_rand.append(all_mean_100_rewards[0])
        overall_acc_4_rand.append(all_mean_100_rewards[1])
        overall_acc_20_rand.append(all_mean_100_rewards[2])
        overall_acc_60_rand.append(all_mean_100_rewards[3])

    acc_0_init_np = np.array(overall_acc_0_init)
    acc_4_init_np = np.array(overall_acc_4_init)
    acc_20_init_np = np.array(overall_acc_20_init)
    acc_60_init_np = np.array(overall_acc_60_init)

    acc_0_rand_np = np.array(overall_acc_0_rand)
    acc_4_rand_np = np.array(overall_acc_4_rand)
    acc_20_rand_np = np.array(overall_acc_20_rand)
    acc_60_rand_np = np.array(overall_acc_60_rand)

    acc_0_init_mean = np.mean(acc_0_init_np, axis=0)
    acc_4_init_mean = np.mean(acc_4_init_np, axis=0)
    acc_20_init_mean = np.mean(acc_20_init_np, axis=0)
    acc_60_init_mean = np.mean(acc_60_init_np, axis=0)

    acc_0_rand_mean = np.mean(acc_0_rand_np, axis=0)
    acc_4_rand_mean = np.mean(acc_4_rand_np, axis=0)
    acc_20_rand_mean = np.mean(acc_20_rand_np, axis=0)
    acc_60_rand_mean = np.mean(acc_60_rand_np, axis=0)

    all_acc_mean = [acc_0_init_mean, acc_4_init_mean, acc_20_init_mean, acc_60_init_mean]

    plt.clf()
    for acc, lbl in zip(all_acc_mean, lbls):
        plt.plot(np.arange(len(acc)), acc, label=lbl)
    plt.legend(title="Pruning (%):")
    plt.xlabel("Iteration")
    plt.ylabel("Mean of 100 last rewards")
    plt.savefig("lotteryticket_rl_smart_init_mean")
    plt.close()

    all_acc_mean = [acc_0_rand_mean, acc_4_rand_mean, acc_20_rand_mean, acc_60_rand_mean]

    plt.clf()
    for acc, lbl in zip(all_acc_mean, lbls):
        plt.plot(np.arange(len(acc)), acc, label=lbl)
    plt.legend(title="Pruning (%):")
    plt.xlabel("Iteration")
    plt.ylabel("Mean of 100 last rewards")
    plt.savefig("lotteryticket_rl_rand_init_mean")
    plt.close()






    # plt.plot(np.arange(len(reward_100)), reward_100)
    # plt.plot(np.arange(len(reward_100)), [195]*len(reward_100))
    # plt.title("b_s = 64, g = 0.95, e_d = 0.995, mse \n e_m = 0.01, l_r = 0.001, m=10^4, 2hl(24)")
    # plt.savefig("dqn8.png")

    # plots = [loss_plt, rewards, reward_100]
    # titles = ["loss", "rewards", "mean_100_rewards"]
    # plt.clf()
    # plt.figure()
    # for i in range(3):
    #     plt.subplot(1, 3, i + 1)
    #     plt.plot(np.arange(len(plots[i])), plots[i])
    #     plt.title(titles[i])
    # plt.suptitle("Cartpole (DQN)")
    # plt.savefig("cpdqn_target_lenet")
    # plt.close('all')




main()