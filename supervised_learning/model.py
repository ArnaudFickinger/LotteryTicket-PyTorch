###
'''
June 2019
Code by: Arnaud Fickinger
'''
###

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

class Lenet(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim):
        super(Lenet, self).__init__()
        self.fc1 = nn.Linear(input_dim, h1_dim)
        # print("w before init")
        # print(self.fc1.weight.data)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        # print("w after init")
        # print(self.fc1.weight.data)
        torch.nn.init.zeros_(self.fc1.bias)
        self.fc1_w_init = self.fc1.weight.data
        self.fc1_b_init = self.fc1.bias.data
        # print("iw at init")
        # print(self.fc1_w_init)
        self.mask1 = nn.Linear(input_dim, h1_dim)
        torch.nn.init.ones_(self.mask1.weight)
        torch.nn.init.ones_(self.mask1.bias)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        self.fc2_w_init = self.fc2.weight.data
        self.fc2_b_init = self.fc2.bias.data
        self.mask2 = nn.Linear(h1_dim, h2_dim)
        torch.nn.init.ones_(self.mask2.weight)
        torch.nn.init.ones_(self.mask2.bias)
        self.fc3 = nn.Linear(h2_dim, 10)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)
        self.fc3_w_init = self.fc3.weight.data
        self.fc3_b_init = self.fc3.bias.data
        self.mask3 = nn.Linear(h2_dim, 10)
        torch.nn.init.ones_(self.mask3.weight)
        torch.nn.init.ones_(self.mask3.bias)

    def forward(self, x, verbal = False):
        x = x.view(-1, 28*28)
        # if verbal:
        #     print("w before mask")
        #     print(self.fc1.weight.data)
        #     print(torch.abs(self.fc1.weight.data).sum())
        self.fc1.weight.data = torch.mul(self.fc1.weight, self.mask1.weight)
        # if verbal:
        #     print("w after mask")
        #     print(torch.abs(self.fc1.weight.data).sum())
        #     print(self.fc1.weight.data)
        self.fc2.weight.data = torch.mul(self.fc2.weight, self.mask2.weight)
        self.fc3.weight.data = torch.mul(self.fc3.weight, self.mask3.weight)

        self.fc1.bias.data = torch.mul(self.fc1.bias, self.mask1.bias)
        self.fc2.bias.data = torch.mul(self.fc2.bias, self.mask2.bias)
        self.fc3.bias.data = torch.mul(self.fc3.bias, self.mask3.bias)

        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = self.fc3(h2)
        return F.softmax(h3)

    def reset_mask(self):
        # print("mask1 before reset")
        # print(self.mask1.weight.data)
        self.mask1.weight.data = torch.ones_like(self.mask1.weight)
        # print("mask1 after reset")
        # print(self.mask1.weight.data)
        self.mask2.weight.data = torch.ones_like(self.mask2.weight)
        self.mask3.weight.data = torch.ones_like(self.mask3.weight)
        self.mask1.bias.data = torch.ones_like(self.mask1.bias)
        self.mask2.bias.data = torch.ones_like(self.mask2.bias)
        self.mask3.bias.data = torch.ones_like(self.mask3.bias)

    def save_trained_weight(self):
        self.fc1_w_trained = self.fc1.weight.data
        self.fc2_w_trained = self.fc2.weight.data
        self.fc3_w_trained = self.fc3.weight.data
        self.fc1_b_trained = self.fc1.bias.data
        self.fc2_b_trained = self.fc2.bias.data
        self.fc3_b_trained = self.fc3.bias.data

    def random_reinit(self):
        # print("w before rand")
        # print(self.fc1.weight.data)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        # print("w after rand")
        # print(self.fc1.weight.data)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        torch.nn.init.xavier_normal_(self.fc3.weight)
        torch.nn.init.zeros_(self.fc3.bias)

    def reinitializ(self):
        # print("w before reinit")
        # print(self.fc1.weight.data)
        self.fc1.weight.data = self.fc1_w_init
        # print("w after reinit")
        # print(self.fc1.weight.data)
        self.fc2.weight.data = self.fc2_w_init
        self.fc3.weight.data = self.fc3_w_init
        self.fc1.bias.data = self.fc1_b_init
        self.fc2.bias.data = self.fc2_b_init
        self.fc3.bias.data = self.fc3_b_init

    def load_trained_weight(self):
        # print("w before load")
        # print(self.fc1.weight.data)
        self.fc1.weight.data = self.fc1_w_trained
        # print("w after load")
        # print(self.fc1.weight.data)
        self.fc2.weight.data = self.fc2_w_trained
        self.fc3.weight.data = self.fc3_w_trained
        self.fc1.bias.data = self.fc1_b_trained
        self.fc2.bias.data = self.fc2_b_trained
        self.fc3.bias.data = self.fc3_b_trained

    def prune(self, rate, rate_output):
        fc1_w_treshold = np.percentile(torch.abs(self.fc1.weight).detach().cpu().numpy(), rate)
        fc2_w_treshold = np.percentile(torch.abs(self.fc2.weight).detach().cpu().numpy(), rate)
        fc3_w_treshold = np.percentile(torch.abs(self.fc3.weight).detach().cpu().numpy(), rate_output)

        # print("mask1 before prune")
        # print(self.mask1.weight.data)

        self.mask1.weight.data = torch.mul(torch.gt(torch.abs(self.fc1.weight), fc1_w_treshold).float(), self.mask1.weight)

        # print("mask1 after prune")
        # print(self.mask1.weight.data)

        self.mask2.weight.data = torch.mul(torch.gt(torch.abs(self.fc2.weight), fc2_w_treshold).float(), self.mask2.weight)
        self.mask3.weight.data = torch.mul(torch.gt(torch.abs(self.fc3.weight), fc3_w_treshold).float(), self.mask3.weight)

        fc1_b_treshold = np.percentile(torch.abs(self.fc1.bias).detach().cpu().numpy(), rate)
        fc2_b_treshold = np.percentile(torch.abs(self.fc2.bias).detach().cpu().numpy(), rate)
        fc3_b_treshold = np.percentile(torch.abs(self.fc3.bias).detach().cpu().numpy(), rate_output)

        self.mask1.bias.data = torch.mul(torch.gt(torch.abs(self.fc1.bias), fc1_b_treshold).float(), self.mask1.bias)
        self.mask2.bias.data = torch.mul(torch.gt(torch.abs(self.fc2.bias), fc2_b_treshold).float(), self.mask2.bias)
        self.mask3.bias.data = torch.mul(torch.gt(torch.abs(self.fc3.bias), fc3_b_treshold).float(), self.mask3.bias)





