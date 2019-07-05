###
'''
June 2019
Code by: Arnaud Fickinger
'''
###

import torch
from model import Lenet

from options import Options
import matplotlib

matplotlib.use('agg')

import matplotlib.pyplot as plt
opt = Options().parse()
import torch.nn.functional as F
import numpy as np

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# def exp_2(): #wich weight are favorised and choose the best weight for 100 experience, can we predict if there will be a winning ticket or which pruning will be the best just seeing the initializationm, try normal init on bias, look how xavier works, being stuck in local minima, create ai model to predict if init good and what pruning should we do, and if we can do it now
#     train_loader = DataLoader(
#         datasets.MNIST('../data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=opt.batch_size, shuffle=True)
#
#     test_loader = DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])),
#         batch_size=opt.batch_size, shuffle=True)
#
#     model = Lenet(28 * 28, opt.h1_dim, opt.h2_dim).to(device)
#
#
#     optimizer = torch.optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=opt.lr)
#
#     all_acc = []
#
#     acc_0 = []
#
#     iteration = 0
#     for epoch in range(1, opt.epochs + 1):
#         print(epoch)
#         model.train()
#         for batch_idx, (data, target) in enumerate(train_loader):
#             iteration += 1
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = F.nll_loss(output, target)
#             loss.backward()
#             optimizer.step()
#             if iteration % opt.record_every == 0:
#
#                 model.eval()
#                 test_loss = 0
#                 correct = 0
#                 with torch.no_grad():
#                     for data, target in test_loader:
#                         data, target = data.to(device), target.to(device)
#                         output = model(data)
#                         # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#                         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#                         correct += pred.eq(target.view_as(pred)).sum().item()
#
#                 # test_loss /= len(test_loader.dataset)
#
#                 acc_0.append(correct / len(test_loader.dataset))
#
#     all_acc.append(acc_0)
#
#
# def exp3(): #change the dataset take half of mnist reinitialize train on the other half, i like projects that mix theory amnd practice, i lik eto think of the mathemtical foundation amd the computational challenges so finding a theory to dl is perfect,
#     pass

def main():

    sum_mw1_after_pruning = []

    train_loader = DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=opt.batch_size, shuffle=True)

    test_loader = DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=opt.batch_size, shuffle=True)

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

        model = Lenet(28*28, opt.h1_dim, opt.h2_dim).to(device)

        optimizer = torch.optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=opt.lr)

        all_acc = []

        acc_0 = []

        iteration = 0
        for epoch in range(1, opt.epochs+1):
            # print(epoch)
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                iteration+=1
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if iteration % opt.record_every == 0:

                    model.eval()
                    # test_loss = 0
                    correct = 0
                    with torch.no_grad():
                        for data, target in test_loader:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                            correct += pred.eq(target.view_as(pred)).sum().item()

                    # test_loss /= len(test_loader.dataset)

                    acc_0.append(correct/len(test_loader.dataset))

        all_acc.append(acc_0)

        model.save_trained_weight()

        print("smart")

        for rate in pruning:
            tmpr_acc = []
            output_rate = int(rate/2)

            optimizer = torch.optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=opt.lr) #try with that
            print("..........")

            print(model.mask1.weight.data.sum().item())

            model.load_trained_weight()
            print(model.mask1.weight.data.sum().item())
            model.reset_mask()
            print(model.mask1.weight.data.sum().item())
            model.prune(rate, output_rate)
            print(model.mask1.weight.data.sum().item())
            model.reinitializ()
            print(model.mask1.weight.data.sum().item())



            print(rate)

            print(model.mask1.weight.data.sum().item())

            print(model.mask1.weight.data.shape)

            print(model.mask1.weight.data)

            print("..........")



            for epoch in range(1, opt.epochs + 1):
                # print(epoch)
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    iteration += 1
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    if iteration % opt.record_every == 0:
                        model.eval()
                        test_loss = 0
                        correct = 0
                        with torch.no_grad():
                            for data, target in test_loader:
                                data, target = data.to(device), target.to(device)
                                output = model(data)
                                # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                                correct += pred.eq(target.view_as(pred)).sum().item()

                        # test_loss /= len(test_loader.dataset)

                        tmpr_acc.append(correct / len(test_loader.dataset))
            all_acc.append(tmpr_acc)

        if opt.plot:

            plt.clf()
            for acc, lbl in zip(all_acc, lbls):
                plt.plot(np.arange(len(acc)), acc, label = lbl)
            plt.legend(title = "Pruning (%):")
            plt.xlabel("Iteration")
            plt.ylabel("Test Accuracy")
            plt.savefig("lotteryticket_smart_init_{}".format(test))
            plt.close()


        overall_acc_0_init.append(all_acc[0])
        overall_acc_4_init.append(all_acc[1])
        overall_acc_20_init.append(all_acc[2])
        overall_acc_60_init.append(all_acc[3])



        all_acc = []
        all_acc.append(acc_0)
        print("rand")

        for rate in pruning:
            tmpr_acc = []
            output_rate = int(rate / 2)

            optimizer = torch.optim.Adam([w for name, w in model.named_parameters() if not 'mask' in name], lr=opt.lr)

            model.load_trained_weight()
            model.reset_mask()
            model.prune(rate, output_rate)
            model.random_reinit()

            print("..........")

            print(rate)

            print(model.mask1.weight.data.sum().item())

            print(model.mask1.weight.data.shape)

            print(model.mask1.weight.data)

            print("..........")





            for epoch in range(1, opt.epochs + 1):
                print(epoch)
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    iteration += 1
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()
                    optimizer.step()
                    if iteration % opt.record_every == 0:
                        model.eval()
                        test_loss = 0
                        correct = 0
                        with torch.no_grad():
                            for data, target in test_loader:
                                data, target = data.to(device), target.to(device)
                                output = model(data)
                                # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                                correct += pred.eq(target.view_as(pred)).sum().item()

                        # test_loss /= len(test_loader.dataset)

                        tmpr_acc.append(correct / len(test_loader.dataset))
            all_acc.append(tmpr_acc)

        if opt.plot:

            plt.clf()
            for acc, lbl in zip(all_acc, lbls):
                plt.plot(np.arange(len(acc)), acc, label=lbl)
            plt.legend(title="Pruning (%):")
            plt.xlabel("Iteration")
            plt.ylabel("Test Accuracy")
            plt.savefig("lotteryticket_rand_init_{}".format(test))
            plt.close()

        overall_acc_0_rand.append(all_acc[0])
        overall_acc_4_rand.append(all_acc[1])
        overall_acc_20_rand.append(all_acc[2])
        overall_acc_60_rand.append(all_acc[3])

    acc_0_init_np = np.array(overall_acc_0_init)
    acc_4_init_np = np.array(overall_acc_4_init)
    acc_20_init_np = np.array(overall_acc_20_init)
    acc_60_init_np = np.array(overall_acc_60_init)

    acc_0_rand_np = np.array(overall_acc_0_rand)
    acc_4_rand_np = np.array(overall_acc_4_rand)
    acc_20_rand_np = np.array(overall_acc_20_rand)
    acc_60_rand_np = np.array(overall_acc_60_rand)

    acc_0_init_mean = np.mean(acc_0_init_np, axis = 0)
    acc_4_init_mean = np.mean(acc_4_init_np, axis=0)
    acc_20_init_mean = np.mean(acc_20_init_np, axis=0)
    acc_60_init_mean = np.mean(acc_60_init_np, axis=0)

    acc_0_rand_mean = np.mean(acc_0_rand_np, axis=0)
    acc_4_rand_mean = np.mean(acc_4_rand_np, axis=0)
    acc_20_rand_mean = np.mean(acc_20_rand_np, axis=0)
    acc_60_rand_mean = np.mean(acc_60_rand_np, axis=0)

    all_acc_mean = [acc_0_init_mean, acc_4_init_mean, acc_20_init_mean, acc_60_init_mean]

    if opt.plot:

        plt.clf()
        for acc, lbl in zip(all_acc_mean, lbls):
            plt.plot(np.arange(len(acc)), acc, label=lbl)
        plt.legend(title="Pruning (%):")
        plt.xlabel("Iteration")
        plt.ylabel("Test Accuracy")
        plt.savefig("lotteryticket_smart_init_mean")
        plt.close()

        all_acc_mean = [acc_0_rand_mean, acc_4_rand_mean, acc_20_rand_mean, acc_60_rand_mean]

        plt.clf()
        for acc, lbl in zip(all_acc_mean, lbls):
            plt.plot(np.arange(len(acc)), acc, label=lbl)
        plt.legend(title="Pruning (%):")
        plt.xlabel("Iteration")
        plt.ylabel("Test Accuracy")
        plt.savefig("lotteryticket_rand_init_mean")
        plt.close()






main()



