# -*- coding: utf-8 -*-
# @Time : 2023/5/13 12:51
# @Author : Wanghong
# @FileName: tantiandata.py
# @Software: PyCharm
import copy
import os

import h5py
import json

import numpy as np
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler

from utils import *
from const import *
import matplotlib.pyplot as plt
import ahrs

Subjects_num = ['subject_'+str(x).rjust(2, '0') for x in range(1, 18)]

IMU_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ']


def map_seg(seg):
    return [field + '_' + seg for field in IMU_FIELDS]


def get_q(array):
    acc = array[:, [0, 1, 2]]
    gyr = array[:, [3, 4, 5]]
    gyr_0 = np.radians(gyr)
    q = ahrs.filters.Mahony(gyr_0, acc, frequency=100, k_P=1, k_I=0.01)
    return np.hstack((acc, gyr, q.Q))


def feature_ext(data, use_feature=False, use_raw_data=True):
    """extract norm, mean features of IMUs"""
    if use_feature:
        for i in range(len(sensor_list)):
            acc1_norm = np.linalg.norm(data[:, i * len(IMU_FIELDS):i * len(IMU_FIELDS) + 3], axis=1)
            gyr1_norm = np.linalg.norm(data[:, i * len(IMU_FIELDS) + 3:i * len(IMU_FIELDS) + 6], axis=1)
            acc1_mean = np.mean(data[:, i * len(IMU_FIELDS):i * len(IMU_FIELDS) + 3], axis=1)
            gyr1_mean = np.mean(data[:, i * len(IMU_FIELDS) + 3:i * len(IMU_FIELDS) + 6], axis=1)

            feature = [acc1_norm, gyr1_norm, acc1_mean, gyr1_mean]
            feature = np.array(feature)
            if use_raw_data:
                data = np.insert(data, data.shape[1], feature, axis=1)
            else:
                data = feature
    else:
        pass
    return data


def scale_data(train_data, valid_data, test_data):
    scale_x = MinMaxScaler(feature_range=(-1, 1))
    train_dataset_list = []
    for i in train_data:
        train_dataset_list += list(i)
    train_dataset_array = np.asarray(train_dataset_list)
    scale_x.fit(train_dataset_array)
    for i in range(len(train_data)):
        train_data[i] = scale_x.transform(train_data[i])
    for i in range(len(valid_data)):
        valid_data[i] = scale_x.transform(valid_data[i])
    for i in range(len(test_data)):
        test_data[i] = scale_x.transform(test_data[i])
    return train_data, valid_data, test_data


def generate_tantiandata(tantiandata_path, hyper_train_sub_ids, hyper_vali_sub_ids, test_sub_ids):
    with h5py.File(tantiandata_path, 'r') as hf:
        data_all_sub = {subject: subject_data[:] for subject, subject_data in hf.items()}
        data_fields = json.loads(hf.attrs['columns'])

    train_data, valid_data, test_data, train_label, valid_label, test_label = [], [], [], [], [], []
    train_lens, valid_lens, test_lens = [], [], []
    train_subs, valid_subs, test_subs = [], [], []
    # average_stride_width_initial_double_support_each_subject = []

    for subject_id, subject_num in zip(range(17), Subjects_num):

        data_sub = data_all_sub[subject_num]  # 3d array (B, S=152, F=256)
        each_step_data = {'imu_step': [], 'r_sw_step': [],
                          'lens': [], 'subs': []}
        for step in range(len(data_sub)):
            data_sub_step = np.empty((152, 1))
            data_sub_step_ = data_sub[step, :, :]
            # find HS-TO
            body_weight_loc = data_fields.index('body weight')
            body_weight_index = np.where(data_sub_step_[:, body_weight_loc] == 0)[0]
            lens = body_weight_index[0]-20-20
            data_sub_step_HS_TO_padded = np.zeros(shape=data_sub_step_.shape)
            data_sub_step_HS_TO_padded[:lens, :] = data_sub_step_[20:body_weight_index[0]-20, :]
            each_step_data['lens'].append(lens)
            each_step_data['subs'].append(subject_id)
            # data_sub_step_HS_TO_padded = data_sub_step_

            for seg in ['R_SHANK', 'WAIST', 'L_SHANK']:
                imu_seg_loc = [data_fields.index(field) for field in map_seg(seg)]
                data_sub_seg = data_sub_step_HS_TO_padded[:, imu_seg_loc]
                # data_sub_seg = get_q(data_sub_seg)
                data_sub_step = np.hstack((data_sub_step, data_sub_seg))
            data_sub_step = data_sub_step[:, 1:]
            data_sub_step = data_filter(data_sub_step, 20, 100)
            imu_step_data = feature_ext(data_sub_step, use_feature=True)

            middle_index = 1 + 20
            # right_ankle_loc, left_ankle_loc = data_fields.index('RAnkle_x_180'), data_fields.index('LAnkle_x_180')
            left_ankle_loc, right_ankle_loc = data_fields.index('LFCC_X'), data_fields.index('RFCC_X')
            stride_width = (data_sub_step_[middle_index, right_ankle_loc] - data_sub_step_[middle_index, left_ankle_loc]) / 10

            each_step_data['imu_step'].append(imu_step_data)
            each_step_data['r_sw_step'].append(stride_width)
        # print(np.mean(np.array(each_step_data['r_sw_step'])))

        if subject_id in hyper_train_sub_ids:
            train_data += each_step_data['imu_step']
            train_label += each_step_data['r_sw_step']
            train_lens += each_step_data['lens']
            train_subs += each_step_data['subs']
        if subject_id in hyper_vali_sub_ids:
            valid_data += each_step_data['imu_step']
            valid_label += each_step_data['r_sw_step']
            valid_lens += each_step_data['lens']
            valid_subs += each_step_data['subs']
        if subject_id in test_sub_ids:
            test_data += each_step_data['imu_step']
            test_label += each_step_data['r_sw_step']
            test_lens += each_step_data['lens']
            test_subs += each_step_data['subs']

    return train_data, train_label, train_lens, train_subs, valid_data, valid_label, valid_lens, valid_subs, test_data, test_label, test_lens, test_subs


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
        """每一步，对应一个增强"""
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

        # theta_x = np.random.uniform(-5, 5)
        # theta_y = np.random.uniform(-30, 30)
        # theta_z = np.random.uniform(-10, 10)
        # R = rotate.rotate_around_xyz(theta_x, theta_y, theta_z)
        arg_train_data[i] = rotate.rotate_data_left_shank(R)
        if scalar:
            arg_train_data[i] = scalar.transform(arg_train_data[i])

        arg_train_label[i] = train_label[i]
        arg_train_lens[i] = train_lens[i]

    if not keep_rawdata:
        return arg_train_data, arg_train_label, arg_train_lens
    return np.concatenate((train_data, arg_train_data), axis=0), np.concatenate((train_label, arg_train_label), axis=0), np.concatenate((train_lens, arg_train_lens), axis=0)


def augmentation_time_wrap(train_data, train_label, train_lens, keep_rawdata):
    from data_augmentation import DA_TimeWarp
    arg_train_data = np.zeros(shape=train_data.shape)
    arg_train_label = np.zeros(shape=train_label.shape)
    arg_train_lens = np.zeros(shape=train_lens.shape)
    for i in range(len(train_data)):
        """每一步，对应一个增强"""
        arg_train_data[i] = DA_TimeWarp(train_data[i])
        arg_train_label[i] = train_label[i]
        arg_train_lens[i] = train_lens[i]
    if not keep_rawdata:
        return arg_train_data, arg_train_label, arg_train_lens
    return np.concatenate((train_data, arg_train_data), axis=0), np.concatenate((train_label, arg_train_label), axis=0), np.concatenate((train_lens, arg_train_lens), axis=0)


def sequence_tan(hyper_train_sub_ids, hyper_vali_sub_ids, test_sub_ids):
    max_length = 152
    train_data, train_label, train_lens, train_subs, valid_data, valid_label, valid_lens, valid_subs, test_data, test_label, test_lens, test_subs = generate_tantiandata(tantiandata_path, hyper_train_sub_ids, hyper_vali_sub_ids, test_sub_ids)

    augmentation_flag = False
    if augmentation_flag:
        # setup_seed(42)
        d, e, f = augmentation_time_wrap(np.array(train_data), np.array(train_label), np.array(train_lens), keep_rawdata=False)
        a, b, c = augmentation(np.array(train_data), np.array(train_label), np.array(train_lens), None, keep_rawdata=False)

        new_train_data = np.concatenate((train_data, d, a), axis=0)
        new_train_label = np.concatenate((train_label, e, b), axis=0)
        new_train_lens = np.concatenate((train_lens, f, c), axis=0)
        new_train_subs = np.concatenate((train_subs, train_subs, train_subs), axis=0)

        # revision
        # reduce training data effects, training dataset decreased by 10%, 20%, 30%, 40%, 50%
        # first, decreased by 20%
        # length = len(new_train_data)
        # indices = list(range(length))
        # random.shuffle(indices)
        # indices = indices[:int(length * 0.9)]
        # new_train_data = new_train_data[indices]
        # new_train_label = new_train_label[indices]
        # new_train_lens = new_train_lens[indices]
        # new_train_subs = new_train_subs[indices]

        new_train_data, valid_data, test_data = scale_data(new_train_data, valid_data, test_data)
        return new_train_data, new_train_label, new_train_lens, new_train_subs, valid_data, valid_label, valid_lens, valid_subs, test_data, test_label, test_lens, test_subs, max_length
    else:
        train_data, valid_data, test_data = scale_data(train_data, valid_data, test_data)
        return train_data, train_label, train_lens, train_subs, valid_data, valid_label, valid_lens, valid_subs, test_data, test_label, test_lens, test_subs, max_length





