import copy
import json
import os

import matplotlib.pyplot as plt
from scipy.stats import shapiro
import plotly.graph_objs as go
import random
import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from model import RNN
from torch.utils.data import DataLoader
from load_data import load_data
from my_log import MyLog
from train_eval import run_model
from utils import setup_seed
from model import Sharifi_CNN

"""Step width estimation for individuals with ataxia"""

subject_info = pd.read_excel('../peter_Israel/Sub Info.xlsx', 'Sheet1')
subject_info.set_index(subject_info.columns[0], inplace=True)
gaitmat_subjects_new = ['26062023', '28062023', '0507202310', '0507202311', '1207202310', '1207202311', '2407202313',
                        '2607202310', '2607202311', '3107202314', '3107202315', '2407202314']
MAX_BUFFER_LEN = 152
log_path = 'log'
pretrained_model_path = 'saved_model/tan/RNN_without_q_FOR_SAGE/model_FOR_SAGE.pth'
pretrained_scaler_path = 'saved_model/tan/RNN_without_q_FOR_SAGE/healthy_min_max_scalar_without_q_for_sage.pkl'
device = torch.device("cuda")

test_set_sub_num = 2
log = MyLog(log_path, "Step Width Estimation")

LOSS_test = []

myEvaluation_each_sub = []
fold_models = []
test_sub_list = []
folder_num = int(np.floor(len(gaitmat_subjects_new) / test_set_sub_num))


class rotate_acc_gyro():
    def __init__(self, data):
        """data:{k, p, q}"""
        self.data = copy.deepcopy(data)
        self.new_data = np.zeros(shape=data.shape)

    def rotate_around_x(self, theta):
        theta = np.radians(theta)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta), -np.sin(theta)],
                       [0, np.sin(theta), np.cos(theta)]])
        return Rx

    def rotate_around_y(self, theta):
        theta = np.radians(theta)
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
        return Ry

    def rotate_around_z(self, theta):
        theta = np.radians(theta)
        Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
        return Rz

    def rotate_around_xyz(self, thetax, thetay, thetaz):
        theta_x = np.radians(thetax)
        theta_y = np.radians(thetay)
        theta_z = np.radians(thetaz)

        # Create the rotation matrices for each axis
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(theta_x), -np.sin(theta_x)],
                       [0, np.sin(theta_x), np.cos(theta_x)]])

        Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                       [0, 1, 0],
                       [-np.sin(theta_y), 0, np.cos(theta_y)]])

        Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                       [np.sin(theta_z), np.cos(theta_z), 0],
                       [0, 0, 1]])

        # Combine the rotations
        R = Rz @ Ry @ Rx
        return R

    def rotate_data_right_shank(self, R):
        for k in range(len(self.data)):
            acc1 = self.data[k][:, 0:3]
            gyro1 = self.data[k][:, 3:6]
            self.new_data[k][:, 0:3] = acc1 @ R.T
            self.new_data[k][:, 3:6] = gyro1 @ R.T
            self.new_data[k][:, 18] = np.linalg.norm(acc1, axis=1)
            self.new_data[k][:, 19] = np.linalg.norm(gyro1, axis=1)
            self.new_data[k][:, 20] = np.mean(acc1, axis=1)
            self.new_data[k][:, 21] = np.mean(gyro1, axis=1)
        return self.new_data

    def rotate_data_pelvis(self, R):
        for k in range(len(self.data)):
            acc2 = self.data[k][:, 6:9]
            gyro2 = self.data[k][:, 9:12]
            self.new_data[k][:, 6:9] = acc2 @ R.T
            self.new_data[k][:, 9:12] = gyro2 @ R.T
            self.new_data[k][:, 22] = np.linalg.norm(acc2, axis=1)
            self.new_data[k][:, 23] = np.linalg.norm(gyro2, axis=1)
            self.new_data[k][:, 24] = np.mean(acc2, axis=1)
            self.new_data[k][:, 25] = np.mean(gyro2, axis=1)
        return self.new_data

    def rotate_data_left_shank(self, R):
        for k in range(len(self.data)):
            acc3 = self.data[k][:, 12:15]
            gyro3 = self.data[k][:, 15:18]
            self.new_data[k][:, 12:15] = acc3 @ R.T
            self.new_data[k][:, 15:18] = gyro3 @ R.T
            self.new_data[k][:, 26] = np.linalg.norm(acc3, axis=1)
            self.new_data[k][:, 27] = np.linalg.norm(gyro3, axis=1)
            self.new_data[k][:, 28] = np.mean(acc3, axis=1)
            self.new_data[k][:, 29] = np.mean(gyro3, axis=1)

        return self.new_data


def augmentation(train_data, train_label, train_lens, scalar, keep_rawdata):
    arg_train_data = np.zeros(shape=train_data.shape)
    arg_train_label = np.zeros(shape=train_label.shape)
    arg_train_lens = np.zeros(shape=train_lens.shape)
    for i in range(len(train_data)):
        """for each step"""
        theta_x = np.random.uniform(-30, 30)
        theta_y = np.random.uniform(-5, 5)
        theta_z = np.random.uniform(-20, 20)
        rotate = rotate_acc_gyro(train_data[i][np.newaxis, :])
        R = rotate.rotate_around_xyz(theta_x, theta_y, theta_z)
        arg_train_data[i] = rotate.rotate_data_pelvis(R)

        theta_x = np.random.uniform(-5, 5)
        theta_y = np.random.uniform(-30, 30)
        theta_z = np.random.uniform(-10, 10)
        R = rotate.rotate_around_xyz(theta_x, theta_y, theta_z)
        arg_train_data[i] = rotate.rotate_data_right_shank(R)

        arg_train_data[i] = rotate.rotate_data_left_shank(R)
        if scalar:
            arg_train_data[i] = scalar.transform(arg_train_data[i])

        arg_train_label[i] = train_label[i]
        arg_train_lens[i] = train_lens[i]

    if not keep_rawdata:
        return arg_train_data, arg_train_label, arg_train_lens
    return np.concatenate((train_data, arg_train_data), axis=0), np.concatenate((train_label, arg_train_label),
                                                                                axis=0), np.concatenate(
        (train_lens, arg_train_lens), axis=0)


def augmentation_time_wrap(train_data, train_label, train_lens, keep_rawdata):
    from data_augmentation import DA_TimeWarp
    arg_train_data = np.zeros(shape=train_data.shape)
    arg_train_label = np.zeros(shape=train_label.shape)
    arg_train_lens = np.zeros(shape=train_lens.shape)
    for i in range(len(train_data)):
        """for each step"""
        arg_train_data[i] = DA_TimeWarp(train_data[i])
        arg_train_label[i] = train_label[i]
        arg_train_lens[i] = train_lens[i]
    if not keep_rawdata:
        return arg_train_data, arg_train_label, arg_train_lens
    return np.concatenate((train_data, arg_train_data), axis=0), np.concatenate((train_label, arg_train_label),
                                                                                axis=0), np.concatenate(
        (train_lens, arg_train_lens), axis=0)


# @profile
def _main(batch_size, lr, epochs):
    print(batch_size, lr, epochs)
    log.add_message("========================================================================")
    log.add_message("lr: {}, batch_sz: {}, epochs : {}".format(lr, batch_size, epochs))
    myEvaluation = []

    for i_folder in range(folder_num):
        setup_seed(0)
        if i_folder < folder_num - 1:
            test_sub = gaitmat_subjects_new[test_set_sub_num * i_folder:test_set_sub_num * (i_folder + 1)]
        else:
            test_sub = gaitmat_subjects_new[test_set_sub_num * i_folder:]

        test_sub_list.append(i_folder)

        train_subs = copy.deepcopy(gaitmat_subjects_new)
        train_subs = [i for i in train_subs if i not in test_sub]
        valid_sub = random.choice(train_subs)
        # fixme
        train_subs.remove(valid_sub)

        log.add_message("train_subs: {}, vali_sub: {}， test_sub: {}".format(train_subs, valid_sub, test_sub))

        with h5py.File('../peter_Israel/patients_new_zeno.h5', 'r') as hf:
            data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
            data_fields = json.loads(hf.attrs['columns'])

        data_cols = [x + y for y in ['1', '2', '3'] for x in
                     ['AccelX_', 'AccelY_', 'AccelZ_', 'GyroX_', 'GyroY_', 'GyroZ_']] \
                    + [x + y for y in ['1', '2', '3'] for x in ['AccNorm_', 'GyroNorm_', 'AccMean_', 'GyroMean_']]

        data_cols_index = [data_fields.index(field) for field in data_cols]
        label_cols_index = data_fields.index('Backheels_width')
        lens_cols_index = data_fields.index('Step_lens')
        trial_cols_index = data_fields.index('Trial')
        subject_cols_index = data_fields.index('Subject')

        test_data, test_label, test_lens = [], [], []
        test_subs_info = []
        for i in test_sub:
            test_data_i = data_all_sub[i][:, :, data_cols_index]
            test_label_i = data_all_sub[i][:, 0, label_cols_index]
            test_lens_i = data_all_sub[i][:, 0, lens_cols_index]
            test_subs_i = data_all_sub[i][:, 0, subject_cols_index]
            test_trials_i = data_all_sub[i][:, 0, trial_cols_index]

            test_data += list(test_data_i)
            test_label += list(test_label_i)
            test_lens += list(test_lens_i)
            test_subs_info += list(test_subs_i)
            # print(i)

        valid_data, valid_label, valid_lens = data_all_sub[valid_sub][:, :, data_cols_index], data_all_sub[valid_sub][:,
                                                                                              0, label_cols_index], \
        data_all_sub[valid_sub][:, 0, lens_cols_index]
        valid_subs_info = data_all_sub[valid_sub][:, 0, subject_cols_index]

        train_data, train_label, train_lens = [], [], []
        train_subs_info = []
        for i in train_subs:
            train_data_i = data_all_sub[i][:, :, data_cols_index]
            train_label_i = data_all_sub[i][:, 0, label_cols_index]
            train_lens_i = data_all_sub[i][:, 0, lens_cols_index]
            train_subs_i = data_all_sub[i][:, 0, subject_cols_index]
            train_trials_i = data_all_sub[i][:, 0, trial_cols_index]

            d, e, f = augmentation_time_wrap(train_data_i, train_label_i, train_lens_i, keep_rawdata=False)
            #
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
        for i in range(len(valid_data)):
            valid_data[i] = scale.transform(valid_data[i])
        for i in range(len(test_data)):
            test_data[i] = scale.transform(test_data[i])

        train_set, valid_set, test_set = load_data(train_data, train_label, train_lens, train_subs_info, valid_data,
                                                   valid_label,
                                                   valid_lens, valid_subs_info, test_data, test_label, test_lens,
                                                   test_subs_info)

        train_iterator = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_iterator = DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
        test_iterator = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

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
                "i_folder": i_folder,
                "test_set": test_subs_info,
                "plot_path": log.dic_path + '/' + str(i_folder) + 'ground_truth_vs_prediction'}

        LOSS_valid = []
        LOSS_train = []

        log.save()

        Temp_accuracy = []

        for epoch in range(1, epochs + 1):
            args["epoch"] = epoch

            train_loss = run_model(model, train_iterator, optimizer, loss_function, args, type="train")

            if epoch % 50 == 0:
                print("Epoch {} complete! Average Training loss cm: {:.4f} ".format(epoch, train_loss))
            log.add_message("Epoch: {} | Training loss cm: {:.4f}".format(epoch, train_loss))
            LOSS_train.append(train_loss)

            # validation
            valid_loss = run_model(model, valid_iterator, optimizer, loss_function, args, type="eval")
            if epoch % 50 == 0:
                print("Epoch {} complete! Average Validation loss cm: {:.4f} ".format(epoch, valid_loss))
            LOSS_valid.append(valid_loss)
            # LOSS_valid_power.append(valid_power_loss / valid_total)
            log.add_message("Epoch: {} | Valid loss cm: {:.4f} ".format(epoch, valid_loss))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(LOSS_train))), y=LOSS_train, mode='lines', name='Train Loss'))
            fig.add_trace(go.Scatter(x=list(range(len(LOSS_valid))), y=LOSS_valid, mode='lines', name='Valid Loss'))
            fig.write_html(log.dic_path + '/' + str(i_folder) + '_loss_curve.html', auto_open=False)

            Use_epochs = list(range(146, 151))

            if epoch in Use_epochs:
                test_loss, R2, RMSE, MAPE, MAE, mean_error, pearsonr_cc, spearmanr_cc, icc, NAPE = run_model(model,
                                                                                                             test_iterator,
                                                                                                             optimizer,
                                                                                                             loss_function,
                                                                                                             args,
                                                                                                             type="test",
                                                                                                             plot=True)

                LOSS_test.append(test_loss)
                Temp_accuracy.append([R2, RMSE, MAPE, MAE, mean_error, pearsonr_cc, spearmanr_cc, icc, NAPE])

                if len(Temp_accuracy) == 5:
                    myEvaluation.append(np.array(Temp_accuracy))
                    Temp_accuracy = []

    myEvaluation_array = np.array(myEvaluation)

    Temp_myEvaluation_array = myEvaluation_array.reshape(-1, 9)
    log.add_csv([["R2", "RMSE", "MAPE", "MAE", "mean_error", "pearsonr_cc", "spearmanr_cc", "ICC", "NAPE"]],
                "myEvaluation")
    log.add_csv(Temp_myEvaluation_array, "myEvaluation")

    log.save()


if __name__ == '__main__':
    batch_size = 48
    lr = 0.0001
    epochs = 150
    _main(batch_size, lr, epochs=150)

