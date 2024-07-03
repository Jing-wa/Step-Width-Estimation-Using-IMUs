# -*- coding: utf-8 -*-
# @Time : 2023/6/25 9:42
# @Author : Wanghong
# @FileName: for_sage.py
# @Software: PyCharm
import json
from copy import deepcopy

import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import joblib
from tantiandata import sequence_tan
from train_eval import *
from my_log import *
from const import *
from load_data import *
from model import *


def for_healthy():
    epochs = 30
    batch_size = 128
    lr = 0.001

    setup_seed(42)
    log = MyLog(tanlog_path, "Trained model in healthy subjects")
    log.add_message("lr: {}, batch_sz: {}, epochs : {}".format(lr, batch_size, epochs))

    sub_ids = shuffle(list(range(17)))

    LOSS_test = []
    fold_models = []
    i_folder = 0

    print("generate train_valid_test data")
    train_data, train_label, train_lens, train_subs, valid_data, valid_label, valid_lens, valid_subs, test_data, test_label, test_lens, test_subs, max_length = sequence_tan(
        list(range(17)), [], [])
    train_set, valid_set, test_set = load_data(train_data, train_label, train_lens, train_subs, valid_data, valid_label, valid_lens, valid_subs, test_data, test_label, test_lens, test_subs)

    train_iterator = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    model = RNN(max_length=max_length, dropout=0.2).to(device)

    loss_function = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    log.add_message('RNN + MAE + FOR SAGE + ALL 17 SUBJECTS')

    args = {"data_max_": None,
            "data_min_": None,
            "plot_path": log.dic_path + '/' + str(i_folder) + 'ground_truth_vs_prediction'}
    log.save()
    best_loss = float('inf')
    epochs_without_improvement = 0
    max_epochs_without_improvement = 10
    for epoch in range(epochs):
        # train
        train_loss = run_model(model, train_iterator, optimizer, loss_function, args, type="train")

        print("Epoch {} complete! Average Training loss cm: {:.4f} ".format(epoch, train_loss))
        log.add_message("Epoch: {} | Training loss cm: {:.4f}".format(epoch, train_loss))

    # save model from each fold
    torch.save(deepcopy(model.state_dict()), os.path.join("trained_models/healthy", "model_FOR_SAGE.pth"))


def for_ataxia():
    from main_ataxia import gaitmat_subjects_new, MAX_BUFFER_LEN, augmentation_time_wrap, augmentation

    epochs = 150
    batch_size = 48
    lr = 0.0001

    setup_seed(0)
    log = MyLog('log', "Trained model in individuals with ataxia")
    log.add_message("lr: {}, batch_sz: {}, epochs : {}".format(lr, batch_size, epochs))

    with h5py.File('../peter_Israel/patients_new_zeno.h5', 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

    data_cols = [x + y for y in ['1', '2', '3'] for x in
                 ['AccelX_', 'AccelY_', 'AccelZ_', 'GyroX_', 'GyroY_', 'GyroZ_']] \
                + [x + y for y in ['1', '2', '3'] for x in ['AccNorm_', 'GyroNorm_', 'AccMean_', 'GyroMean_']]

    data_cols_index = [data_fields.index(field) for field in data_cols]
    label_cols_index = data_fields.index('Backheels_width')
    lens_cols_index = data_fields.index('Step_lens')
    subject_cols_index = data_fields.index('Subject')

    train_data, train_label, train_lens = [], [], []
    train_subs_info = []
    for i in gaitmat_subjects_new:
        train_data_i = data_all_sub[i][:, :, data_cols_index]
        train_label_i = data_all_sub[i][:, 0, label_cols_index]
        train_lens_i = data_all_sub[i][:, 0, lens_cols_index]
        train_subs_i = data_all_sub[i][:, 0, subject_cols_index]

        d, e, f = augmentation_time_wrap(train_data_i, train_label_i, train_lens_i, keep_rawdata=False)
        a, b, c = augmentation(train_data_i, train_label_i, train_lens_i, None, keep_rawdata=False)

        train_data += list(train_data_i) + list(d) + list(a)
        train_label += list(train_label_i) + list(e) + list(b)
        train_lens += list(train_lens_i) + list(f) + list(c)
        train_subs_info += list(train_subs_i) + list(train_subs_i) + list(train_subs_i)

    # scale
    scale = MinMaxScaler(feature_range=(-1, 1))
    train_dataset_list = []
    for i in train_data:
        train_dataset_list += list(i)
    train_dataset_array = np.asarray(train_dataset_list)
    scale.fit(train_dataset_array)

    for i in range(len(train_data)):
        train_data[i] = scale.transform(train_data[i])

    joblib.dump(scale, os.path.join("trained_models/ataxia", "min_max_scaler.pkl"))

    train_set, valid_set, test_set = load_data(train_data, train_label, train_lens, train_subs_info, [],
                                               [],
                                               [], [], [], [], [],
                                               [])

    train_iterator = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    model = RNN(max_length=MAX_BUFFER_LEN).to(device)
    # model = Sharifi_CNN().to(device)
    # load healthy model weights
    # model.load_state_dict(torch.load(pretrained_model_path))

    # loss_function = nn.MSELoss().to(device)
    loss_function = nn.SmoothL1Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    args = {"data_max_": None,
            "data_min_": None,
            "i_folder": 0,
            # "test_set": test_subs_info,
            "plot_path": log.dic_path + '/' + str(0) + 'ground_truth_vs_prediction'}

    for epoch in range(1, epochs + 1):
        args["epoch"] = epoch

        train_loss = run_model(model, train_iterator, optimizer, loss_function, args, type="train")
        print("Epoch {} complete! Average Training loss cm: {:.4f} ".format(epoch, train_loss))
        log.add_message("Epoch: {} | Training loss cm: {:.4f}".format(epoch, train_loss))

    # save model from each fold
    torch.save(deepcopy(model.state_dict()), os.path.join("trained_models/ataxia", "model_FOR_SAGE.pth"))


if __name__ == '__main__':
    # for_healthy()
    for_ataxia()









