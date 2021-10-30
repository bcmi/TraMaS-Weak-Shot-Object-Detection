import os
import json
import torch
import random
from torch.utils.data.sampler import Sampler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.sgd import SGD
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class SimDataset():
    def __init__(self, filename,is_train=True):
        self.n=80
        self.is_train=is_train
        features_path = os.path.join(filename, 'features.json')
        labels_path = os.path.join(filename, 'labels.json')
        features = torch.from_numpy(np.array(json.load(open(features_path)))).float()
        labels = torch.from_numpy(np.array(json.load(open(labels_path))))
        self.cat_features = [[] for _ in range(self.n)]
        self.labels = [None for _ in range(self.n)]
        self.lens = [0 for _ in range(self.n)]
        self.make_cat_features(features, labels)





    def make_cat_features(self, features, labels):
        for  f, l in zip(features, labels):
            self.cat_features[l].append(f.reshape(1,-1))

        for i in range(self.n):
            self.cat_features[i] = torch.cat(self.cat_features[i],dim=0)
            self.labels[i]=torch.ones(size=(len(self.cat_features[i]),1))*i
            self.lens[i]=self.cat_features[i].shape[0]
        if self.is_train:
            self.cat_features=torch.cat(self.cat_features,dim=0)
            self.labels=torch.cat(self.labels,dim=0)

    def __getitem__(self, indices):
        if not self.is_train:
            return self.cat_features[indices]
        features=[]
        labels=[]
        for i in indices:
            features.append(self.cat_features[i].reshape(1,-1))
            labels.append(self.labels[i].reshape(1,-1))
        return torch.cat(features,dim=0),torch.cat(labels,dim=0)

    def __len__(self):
        return 0

    def list_len(self):
        return self.lens

class SimInfSampler(Sampler):
    def __init__(self,dataset):
        self.n=dataset.n

    def __iter__(self):
        return iter(list(range(self.n)))

    def __len__(self):
        return  0

class SimSampler(Sampler):
    def __init__(self, dataset):
        self.n = dataset.n
        self.k=8
        self.sizes = dataset.list_len()
        self.indices_size = [0 for _ in range(self.n)]
        self.indices_size[0] = self.sizes[0]
        for i in range(1, self.n):
            self.indices_size[i] = self.indices_size[i - 1] + self.sizes[i]
        self.sim_inices = []
        self.cur_idx = [0 for _ in range(self.n)]
        self.full = [False for _ in range(self.n)]
        self.full_num = 0
        self.shuffle_indices = [list(range(0, self.sizes[0]))]
        self.shuffle_indices += [list(range(self.indices_size[i - 1], self.indices_size[i])) for i in range(1, self.n)]
        [random.shuffle(self.shuffle_indices[i]) for i in range(self.n)]
        self.make_sim_indices()

    def __iter__(self):
        return iter(self.sim_inices)

    def __len__(self):
        return sum(self.sizes)

    def get_idx(self, idx):
        while self.full[idx]:
            idx += 1
            idx %= self.n
        return idx

    def make_sim_indices(self):
        while True:
            if self.full_num == self.n:
                break
            cat_idx = random.sample(range(0, self.n), self.k)
            sim_indixes = []
            for idx in cat_idx:
                if self.full_num == self.n:
                    break
                idx = self.get_idx(idx)
                if self.full[idx]:
                    idx = self.get_idx(idx)
                elif self.cur_idx[idx] + self.k >= self.sizes[idx]:
                    self.full[idx] = True
                    self.full_num += 1
                    self.cur_idx[idx] += self.k
                    continue
                sim_indixes += self.shuffle_indices[idx][self.cur_idx[idx]:self.cur_idx[idx] + self.k]
                self.cur_idx[idx] += self.k
            if sim_indixes:
                self.sim_inices.append(sim_indixes)


class SimNet(nn.Module):
    def __init__(self, inchannel):
        super().__init__()
        self.sim = nn.Sequential(
            nn.Linear(inchannel*2, inchannel * 4),
            nn.BatchNorm1d(inchannel * 4),
            nn.ReLU(),
            nn.Linear(inchannel * 4, inchannel * 4),
            nn.BatchNorm1d(inchannel * 4),
            nn.ReLU(),
            nn.Linear(inchannel * 4, 1)
        )

    def forward(self, x):
        x = self.sim(x)
        return F.sigmoid(x)

def make_pairwise_features_labels(features,labels):
    _,n,c=features.shape
    features=features.repeat(n,1,1)
    trans_features = features.transpose(0, 1)
    features=features.reshape(-1,c)
    trans_features=trans_features.reshape(-1,c)
    pair_features=torch.cat((trans_features,features),dim=1)

    labels=labels.repeat(n,1,1)
    trans_labels=labels.transpose(0,1)
    labels=labels.reshape(-1,1)
    trans_labels=trans_labels.reshape(-1,1)
    gt_labels=torch.ones(size=(n*n,1))
    for i in range(n*n):
        if labels[i]!=trans_labels[i]:
            gt_labels[i]=0


    return pair_features,gt_labels

def train_simnet(data_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    epoches=10
    train_dataset = SimDataset(data_path)
    train_sampler = SimSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, sampler=train_sampler, num_workers=4)

    model = SimNet(2048)
    model=nn.DataParallel(model)
    model.cuda()
    model.train()

    optimizer = SGD(model.parameters(), lr=1e-5, momentum=0.9)
    criterion = nn.BCELoss()
    for epoch in range(epoches):
        for features, labels in train_dataloader:
            pair_features,labels=make_pairwise_features_labels(features,labels)
            scores = model(pair_features.cuda())
            loss = criterion(scores, labels.cuda())
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(output_path, 'simnet.pth'))

def make_inf_pairwise_features(f,features):
    n=features.shape[1]
    f=f.repeat(1,n,1)
    p_features1=torch.cat((f,features),dim=2)
    p_features2=torch.cat((features,f),dim=2)
    return torch.cat((p_features1,p_features2),dim=1).squeeze(0)

def assign_weights(data_path,checkpoint_path):
    test_dataset = SimDataset(data_path,is_train=False)
    test_sampler=SimInfSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset,sampler=test_sampler, batch_size=1, shuffle=False, num_workers=4)

    model = SimNet(2048)
    model=nn.DataParallel(model)
    model.cuda()
    checkpoint = torch.load(os.path.join(checkpoint_path,'simnet.pth'))
    model.load_state_dict(checkpoint)
    model.eval()
    weights=[]
    for features in test_dataloader:
        for i in range(features.shape[1]):
            f=features[:,i].unsqueeze(1)
            pair_features=make_inf_pairwise_features(f,features)
            scores = model(pair_features.cuda())
            weight = scores.mean(0)
            weights.append(weight.cpu().item())
    json.dump(weights,open(os.path.join(checkpoint_path,'weights.json'),'w'))


if __name__ == '__main__':
    # train_simnet('./','./')
    assign_weights('./','./')