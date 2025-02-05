from dataset import Dataset
from SimplE import SimplE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Trainer:
    def __init__(self, dataset, args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if not args.cont:
            self.model = SimplE(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        else:
            self.model = SimplE(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
            temp = torch.load("models/WN18/1000.chkpnt")
            # self.model.extend(dataset.num_ent(), dataset.num_rel())
            self.model.extend(temp)
        self.dataset = dataset
        self.args = args

    def train(self):
        self.model.train()

        key_ent = [40943, 40944, 40945, 40946, 40947, 25546, 10838, 4207]

        neighbor_ent = [40943, 40944, 40945, 40946, 40947, 25546, 10838, 4207, 25546, 10838, 4207, 1270784, 1338113,
                        1269379, 4151940, 1337224, 1283208, 1362568, 1365131, 260622, 1207951, 1268112, 260881, 248977,
                        1603732, 1271189, 1363482, 1264283, 1335588, 1338663, 265386, 2840619, 1580467, 1337653, 1124794,
                        1336635, 3122748, 1599805, 1363648, 1233993, 265673, 1208400, 1336159, 1268457, 1233387, 1517175,
                        1358328, 1335804]

        neighbor_ent_2 = neighbor_ent + [185857, 1232387, 1360899, 617989, 13272059, 8253450, 958477, 1581070, 3051540,
                                         4453910, 1179167, 712225, 14915622, 3050026, 2840619, 6148148, 266806, 6727224,
                                         122748, 205885, 1261628, 136254, 217152, 13973059, 744004, 3724870, 1265740,
                                         8780881, 1265246, 4187233, 1070708, 1582200, 3691128, 4369025, 5580929, 3380867,
                                         4151940, 251013, 261258, 14828683, 1207951, 10067600, 248977, 3376279, 3530910,
                                         1283746, 3673767, 265386, 800940, 264366, 3817647, 3561657, 1125562, 3357376,
                                         3130563, 3879116, 1261773, 243918, 1266895, 1364184, 4014297, 95971, 3151077,
                                         14992613, 1603303, 3725035, 718573, 10515194, 14789885, 248063, 1266945, 15074568,
                                         13945102, 1123598, 1269008, 2730265, 4605726, 10538272, 1159461, 1494310, 2147109,
                                         258854, 4362025, 1233194, 1362736, 14498096, 14974264, 1125693, 1265989, 2749768,
                                         10397001, 2870092, 4201297, 1603418, 406365, 6196071, 1395049, 4105068, 1581933,
                                         13341052, 7679356, 1580928, 10765189, 1262470, 3875218, 250259, 3285912, 717208,
                                         1267098, 15101854, 14992287, 265119, 321956, 261029, 3366823, 1141160, 1261491,
                                         21939, 5663671, 1124794, 3058107, 3315644, 1126335, 1205696, 265673, 14936010,
                                         3730893, 260051, 3635668, 1300437, 199130, 9257949, 2442205, 718815, 1313249,
                                         4700642, 14627820, 827379, 8050678, 2586619, 1335804]

        optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=0,
            initial_accumulator_value=0.1  # this is added because of the consistency to the original tensorflow code
        )

        for epoch in range(1, self.args.ne + 1):
            last_batch = False
            total_loss = 0.0

            while not last_batch:
                h, r, t, l = self.dataset.next_batch(self.args.batch_size, neg_ratio=self.args.neg_ratio,
                                                     device=self.device)
                if self.args.ud_range == 0:
                    last_batch = self.dataset.was_last_batch()
                    continue
                else:
                    if self.args.ud_range == 1:
                        flag = False
                        h_1 = []
                        r_1 = []
                        t_1 = []
                        l_1 = []
                        for iterat in range(len(h)):
                            if h[iterat] or t[iterat] in key_ent:
                                flag = True
                                h_1.append(h[iterat])
                                r_1.append(r[iterat])
                                t_1.append(t[iterat])
                                l_1.append(l[iterat])
                        h_1 = torch.FloatTensor(h_1).to(self.device).long()
                        r_1 = torch.FloatTensor(r_1).to(self.device).long()
                        t_1 = torch.FloatTensor(t_1).to(self.device).long()
                        l_1 = torch.FloatTensor(l_1).to(self.device)
                        last_batch = self.dataset.was_last_batch()
                        optimizer.zero_grad()
                        scores = self.model(h_1, r_1, t_1)
                        loss = torch.sum(F.softplus(-l_1 * scores)) + (
                                self.args.reg_lambda * self.model.l2_loss() / self.dataset.num_batch(
                            self.args.batch_size))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.cpu().item()
                    elif self.args.ud_range == 3:
                        flag = False
                        h_1 = []
                        r_1 = []
                        t_1 = []
                        l_1 = []
                        for iterat in range(len(h)):
                            if h[iterat] or t[iterat] in neighbor_ent:
                                flag = True
                                h_1.append(h[iterat])
                                r_1.append(r[iterat])
                                t_1.append(t[iterat])
                                l_1.append(l[iterat])
                        h_1 = torch.FloatTensor(h_1).to(self.device).long()
                        r_1 = torch.FloatTensor(r_1).to(self.device).long()
                        t_1 = torch.FloatTensor(t_1).to(self.device).long()
                        l_1 = torch.FloatTensor(l_1).to(self.device)
                        last_batch = self.dataset.was_last_batch()
                        optimizer.zero_grad()
                        scores = self.model(h_1, r_1, t_1)
                        loss = torch.sum(F.softplus(-l_1 * scores)) + (
                                self.args.reg_lambda * self.model.l2_loss() / self.dataset.num_batch(
                            self.args.batch_size))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.cpu().item()
                    elif self.args.ud_range == 4:
                        flag = False
                        h_1 = []
                        r_1 = []
                        t_1 = []
                        l_1 = []
                        for iterat in range(len(h)):
                            if h[iterat] or t[iterat] in neighbor_ent_2:
                                flag = True
                                h_1.append(h[iterat])
                                r_1.append(r[iterat])
                                t_1.append(t[iterat])
                                l_1.append(l[iterat])
                        h_1 = torch.FloatTensor(h_1).to(self.device).long()
                        r_1 = torch.FloatTensor(r_1).to(self.device).long()
                        t_1 = torch.FloatTensor(t_1).to(self.device).long()
                        l_1 = torch.FloatTensor(l_1).to(self.device)
                        last_batch = self.dataset.was_last_batch()
                        optimizer.zero_grad()
                        scores = self.model(h_1, r_1, t_1)
                        loss = torch.sum(F.softplus(-l_1 * scores)) + (
                                self.args.reg_lambda * self.model.l2_loss() / self.dataset.num_batch(
                            self.args.batch_size))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.cpu().item()
                    else:
                        last_batch = self.dataset.was_last_batch()
                        optimizer.zero_grad()
                        scores = self.model(h, r, t)
                        loss = torch.sum(F.softplus(-l * scores)) + (
                                    self.args.reg_lambda * self.model.l2_loss() / self.dataset.num_batch(self.args.batch_size))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.cpu().item()

            print("Loss in iteration " + str(epoch) + ": " + str(total_loss) + "(" + self.dataset.name + ")")

            if epoch % self.args.save_each == 0:
                self.save_model(epoch)
            if self.args.ud_range == 0:
                print("Finish iteration " + str(epoch) + "(" + self.dataset.name + ")")

        self.output_params()

    def save_model(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model, directory + str(chkpnt) + ".chkpnt")

    def output_params(self):
        print("Output Parameters...")
        directory = "output/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        ent_h_embs = np.array(self.model.ent_h_embs.weight.data.cpu())
        np.savetxt(directory + "ent_h_embs.txt", ent_h_embs)
        ent_t_embs = np.atleast_1d(self.model.ent_t_embs.weight.data.cpu())
        np.savetxt(directory + "ent_t_embs.txt", ent_t_embs)
        rel_embs = np.atleast_1d(self.model.rel_embs.weight.data.cpu())
        np.savetxt(directory + "rel_embs.txt", rel_embs)
