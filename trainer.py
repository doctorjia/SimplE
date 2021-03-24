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

        neighbor_ent = [25546, 10838, 4207]

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
