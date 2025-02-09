import os
import pandas as pd
import numpy as np
import torch.utils.data as util_data
from torch.utils.data import Dataset
import torch
from collections import Counter

class VirtualAugSamples(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_x) == len(train_y)
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx]}, idx

class SemiAugSamples(Dataset):
    def __init__(self, train_x, train_y, index):
        assert len(train_x) == len(train_y)
        self.train_x = train_x
        self.train_y = train_y
        self.index = index

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx], 'index': self.index[idx]}, idx
    
class ExplitAugSamples(Dataset):
    def __init__(self, train_x, train_x1, train_x2, train_y, remaining_indices):
        assert len(train_y) == len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_y = train_y
        self.indexs = remaining_indices
        
    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'augmentation_1': self.train_x1[idx], 'augmentation_2': self.train_x2[idx], 'label': self.train_y[idx], 'index':self.indexs[idx]}, idx
       


def augmentation_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname + ".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_text1 = train_data[args.augmentation_1].fillna('.').values
    train_text2 = train_data[args.augmentation_2].fillna('.').values
    train_label = train_data[args.label].astype(int).values
    num_samples = len(train_label)
    arr_pred = Counter(np.array(train_label))
    print('真实：', len(arr_pred), arr_pred)

    # all_indices = np.arange(num_samples)
    # train_dataset = ExplitAugSamples(train_text, train_text1, train_text2, train_label, all_indices)
    # train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
    #                                     drop_last=True)


    if args.classes < 50:
        sample_size = int(num_samples * 0.01)
        random_indices = np.random.choice(num_samples, size=sample_size, replace=False)
        sampled_train_text = train_text[random_indices]
        sampled_train_text1 = train_text1[random_indices]
        sampled_train_text2 = train_text2[random_indices]
        sampled_train_label = train_label[random_indices]
        semi_train_dataset = ExplitAugSamples(sampled_train_text, sampled_train_text1, sampled_train_text2, sampled_train_label, random_indices)
        semi_train_loader = util_data.DataLoader(semi_train_dataset, batch_size=args.semi_batch_size, shuffle=True,
                                                 num_workers=4, drop_last=False)

    else:
        sampled_train_text = []
        sampled_train_text1 = []
        sampled_train_text2 = []
        sampled_train_label = []
        random_indices = []
        remaining_indices = np.arange(num_samples)
        for label in np.unique(train_label):
            select_index = remaining_indices[train_label == label][0]
            random_indices.append(select_index)
            sampled_train_text.append(train_text[select_index])
            sampled_train_text1.append(train_text1[select_index])
            sampled_train_text2.append(train_text2[select_index])
            sampled_train_label.append(label)
        semi_train_dataset = ExplitAugSamples(sampled_train_text, sampled_train_text1, sampled_train_text2, sampled_train_label, random_indices)
        semi_train_loader = util_data.DataLoader(semi_train_dataset, batch_size=args.semi_batch_size, shuffle=True,
                                                 num_workers=4, drop_last=False)



    remaining_indices = np.setdiff1d(np.arange(num_samples), random_indices)
    remaining_train_text = train_text[remaining_indices]
    remaining_train_text1 = train_text1[remaining_indices]
    remaining_train_text2 = train_text2[remaining_indices]
    remaining_train_label = train_label[remaining_indices]
    train_dataset = ExplitAugSamples(remaining_train_text, remaining_train_text1, remaining_train_text2,
                                     remaining_train_label, remaining_indices)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                        drop_last=True)

    # arr_MLP = Counter(np.array(remaining_train_label))

    # 1、计算MLP的熵
    MLP_total = sum(arr_pred.values())
    MLP_probabilities = np.array(list(arr_pred.values())) / MLP_total
    MLP_entropy = -np.sum(MLP_probabilities * np.log(MLP_probabilities))
    # 2、计算变异系数
    MLP_mean = np.mean(MLP_probabilities)
    MLP_std = np.std(MLP_probabilities)
    MLP_cv = MLP_std / MLP_mean
    print('真实label的熵与变异系数：', MLP_entropy, MLP_cv)




    return train_loader, semi_train_loader, sampled_train_label, random_indices


def unshuffle_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+".csv"))
    train_text = train_data[args.text].fillna('.').values
    train_label = train_data[args.label].astype(int).values

    train_dataset = VirtualAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)  # 原是shuffle=False
    return train_loader

