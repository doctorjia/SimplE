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
            self.model = torch.load("models/" + self.dataset.name + "/1000.chkpnt")
        self.dataset = dataset
        self.args = args
        
    def train(self):
        self.model.train()

        optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay= 0,
            initial_accumulator_value= 0.1 #this is added because of the consistency to the original tensorflow code
        )

        for epoch in range(1, self.args.ne + 1):
            last_batch = False
            total_loss = 0.0

            while not last_batch:
                h, r, t, l = self.dataset.next_batch(self.args.batch_size, neg_ratio=self.args.neg_ratio, device = self.device)
                last_batch = self.dataset.was_last_batch()
                optimizer.zero_grad()
                scores = self.model(h, r, t)
                loss = torch.sum(F.softplus(-l * scores))+ (self.args.reg_lambda * self.model.l2_loss() / self.dataset.num_batch(self.args.batch_size))
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

            print("Loss in iteration " + str(epoch) + ": " + str(total_loss) + "(" + self.dataset.name + ")")
        
            if epoch % self.args.save_each == 0:
                self.save_model(epoch)

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
        np.savetxt(directory+"ent_h_embs.txt", ent_h_embs)
        ent_t_embs = np.atleast_1d(self.model.ent_t_embs.weight.data.cpu())
        np.savetxt(directory + "ent_t_embs.txt", ent_t_embs)
        rel_embs = np.atleast_1d(self.model.rel_embs.weight.data.cpu())
        np.savetxt(directory + "rel_embs.txt", rel_embs)
