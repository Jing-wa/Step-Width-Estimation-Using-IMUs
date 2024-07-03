
# # load data
# return train_set, valid_set, test_set
import numpy as np
# from generate_train_valid_test_data import Generate_tvt_dataset
from torch.utils.data import Dataset
# from network_variables import *


class LoadDataSet(Dataset):
    def __init__(self, data, labels, lens=None, subs=None):
        self.data = data
        self.labels = labels
        self.lens = lens
        self.subs = subs

    def __getitem__(self, idx):
        if self.subs is not None:
            return [self.data[idx], self.labels[idx], self.lens[idx], self.subs[idx]]
        else:
            if self.lens is not None:
                return [self.data[idx], self.labels[idx], self.lens[idx]]
            else:
                return [self.data[idx], self.labels[idx]]

    def __len__(self):
        return len(self.data)


def load_data(train_data, train_label, train_lens, train_subs, valid_data, valid_label, valid_lens, valid_subs, test_data, test_label, test_lens, test_subs):

    train_set = LoadDataSet(train_data, train_label, train_lens, train_subs)
    if valid_data is not None:
        valid_set = LoadDataSet(valid_data, valid_label, valid_lens, valid_subs)
    else:
        valid_set = None
    test_set = LoadDataSet(test_data, test_label, test_lens, test_subs)
    return train_set, valid_set, test_set


