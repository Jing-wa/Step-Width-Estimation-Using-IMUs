# -*- coding: utf-8 -*-
# @Time : 2023/3/28 20:56
# @Author : Wanghong
# @FileName: utils.py
# @Software: PyCharm
import csv
import os
from os.path import join as pjoin

import pandas as pd
import scipy
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from torch import nn

import pickle
import torch
import numpy as np
import random
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    shuffle(random_state=seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_to_pickle(data, pickle_path, pickle_name):
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)
    pickle_name = pjoin(pickle_path, pickle_name)
    pickle.dump(data, open(pickle_name, "wb"))
    # print("data saved to a pickle")


def read_from_pickle(pickle_name):
    return pickle.load(open(pickle_name, "rb"))


def finder(pattern, root='.'):
    """找到包含某一字符串的路径"""
    matches = []
    dirs = []
    for x in os.listdir(root):
        if x.startswith('~$'):
            continue  # Skip temporary file
        nd = os.path.join(root, x)
        if os.path.isdir(nd):
            dirs.append(nd)
        elif os.path.isfile(nd) and pattern in x:
            matches.append(nd)
    # for match in matches:
    #     print(match)
    # for dir in dirs:
    #     finder(pattern, root=dir)

    return matches


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        if x.shape != 1:
            x = x.ravel()
        if y.shape != 1:
            y = y.ravel()
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def scale_inverse(data, data_max, data_min, feature_range_max=1, feature_range_min=-1):
    ik_feature_scaled = (data - feature_range_min) / (feature_range_max - feature_range_min)
    ik_scaled = ik_feature_scaled * (data_max - data_min) + data_min
    return ik_scaled


def data_filter(data, cut_off_fre, sampling_fre, filter_order=4):
    fre = cut_off_fre / (sampling_fre / 2)
    b, a = butter(filter_order, fre, 'lowpass')
    if len(data.shape) == 1:
        data_filtered = filtfilt(b, a, data)
    else:
        data_filtered = filtfilt(b, a, data, axis=0)
    return data_filtered


def feature_ext(data, use_feature=False, use_raw_data=True):
    """extract norm, mean features of IMUs"""
    sensor_list = ['Right_foot', 'Lumbar', 'Left_foot'
                   # 'Left_wrist', 'Right_wrist',
                   ]
    measure_list = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'
                    # 'mag_x', 'mag_y', 'mag_z', 'barom', , , 'q_w', 'q_x', 'q_y', 'q_z'
                    ]
    if use_feature:
        for i in range(len(sensor_list)):
            acc1_norm = np.linalg.norm(data[:, i * len(measure_list):i * len(measure_list) + 3], axis=1)
            gyr1_norm = np.linalg.norm(data[:, i * len(measure_list) + 3:i * len(measure_list) + 6], axis=1)
            acc1_mean = np.mean(data[:, i * len(measure_list):i * len(measure_list) + 3], axis=1)
            gyr1_mean = np.mean(data[:, i * len(measure_list) + 3:i * len(measure_list) + 6], axis=1)

            feature = [acc1_norm, gyr1_norm, acc1_mean, gyr1_mean]
            feature = np.array(feature)
            if use_raw_data:
                data = np.insert(data, data.shape[1], feature, axis=1)
            else:
                data = feature
    else:
        pass
    return data


import torch
import os


def save_model(model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint = {
        'model_state_dict': model.state_dict()
    }

    save_file = os.path.join(save_path, 'model_early_stop.pth')
    torch.save(checkpoint, save_file)
    print("Model saved at:", save_file)


import datetime
def calculate_localtime_from_epochs(epoch1, epoch2):
    # if epoch in microseconds (微秒), epoch = 1687778445206880
    time1 = datetime.datetime.fromtimestamp(epoch1/1000/1000)
    time2 = datetime.datetime.fromtimestamp(epoch2/1000/1000)
    return 1 / (time2 - time1)

def pad_zero(data_array, step_length, max_length):
    _shape = [max_length, data_array.shape[1]]
    array_max_length = np.zeros(shape=_shape)
    array_max_length[:step_length, :] = data_array
    return array_max_length


import torch.nn.functional as F

class clustering_loss(nn.Module):
    def __init__(self):
        super(clustering_loss, self).__init__()
    """
    计算聚拢效果的loss。
    参数:
        data (torch.Tensor): 数据张量，大小为 (batch_size, feature_size)

    返回:
        loss (torch.Tensor): 聚拢效果的loss
    """
    def forward(self, data, mean=None):
        if not mean:
            # 计算数据的均值
            mean = torch.mean(data, dim=0, keepdim=True)
            # print(f"mean bias is {mean}")

        # 计算每个数据点到均值的差异
        diff = data - mean

        # 计算loss为差异的平方和（均方误差）
        loss = F.mse_loss(diff, torch.zeros_like(diff))

        return loss


# from calibration import qvq_fun

from ahrs.filters import Mahony
from ahrs.common.orientation import q2R


import numpy as np
from ahrs.common.orientation import dcm2quat

def quatProduct(q1, q2):

    vec_1 = q1[:, 0] * q2[:, 1] + q2[:, 0] * q1[:, 1] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    vec_2 = q1[:, 0] * q2[:, 2] + q2[:, 0] * q1[:, 2] + q1[:, 3] * q2[:, 1] - q1[:, 1] * q2[:, 3]
    vec_3 = q1[:, 0] * q2[:, 3] + q2[:, 0] * q1[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1]

    scalar_0 = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]

    return np.hstack((scalar_0.reshape(-1, 1), vec_1.reshape(-1, 1), vec_2.reshape(-1, 1), vec_3.reshape(-1, 1)))


def qvq_fun(q, v_r):
    q_conj_ = np.hstack((q[:, 0:1], - q[:, 1:4]))
    q_v_r = np.insert(v_r, 0, 0).reshape(-1, 4)
    temp = quatProduct(quatProduct(q, q_v_r), q_conj_)
    v_l = temp[:, 1:4]
    return v_l


def rotMat2quatern(R):
    """
    Converts a rotation matrix orientation to a quaternion.
    """
    R = np.asarray(R)
    numR = R.shape[2] if R.ndim == 3 else 1
    q = np.zeros((numR, 4))
    for i in range(numR):
        K = np.zeros((4, 4))
        R_i = R[..., i] if R.ndim == 3 else R
        K[0, 0] = (1 / 3) * (R_i[0, 0] - R_i[1, 1] - R_i[2, 2])
        K[0, 1] = (1 / 3) * (R_i[1, 0] + R_i[0, 1])
        K[0, 2] = (1 / 3) * (R_i[2, 0] + R_i[0, 2])
        K[0, 3] = (1 / 3) * (R_i[1, 2] - R_i[2, 1])
        K[1, 0] = (1 / 3) * (R_i[1, 0] + R_i[0, 1])
        K[1, 1] = (1 / 3) * (R_i[1, 1] - R_i[0, 0] - R_i[2, 2])
        K[1, 2] = (1 / 3) * (R_i[2, 1] + R_i[1, 2])
        K[1, 3] = (1 / 3) * (R_i[2, 0] - R_i[0, 2])
        K[2, 0] = (1 / 3) * (R_i[2, 0] + R_i[0, 2])
        K[2, 1] = (1 / 3) * (R_i[2, 1] + R_i[1, 2])
        K[2, 2] = (1 / 3) * (R_i[2, 2] - R_i[0, 0] - R_i[1, 1])
        K[2, 3] = (1 / 3) * (R_i[0, 1] - R_i[1, 0])
        K[3, 0] = (1 / 3) * (R_i[1, 2] - R_i[2, 1])
        K[3, 1] = (1 / 3) * (R_i[2, 0] - R_i[0, 2])
        K[3, 2] = (1 / 3) * (R_i[0, 1] - R_i[1, 0])
        K[3, 3] = (1 / 3) * (R_i[0, 0] + R_i[1, 1] + R_i[2, 2])

        eigenvalues, eigenvectors = np.linalg.eig(K)
        # Find the max eigenvalue manually
        max_idx = np.argmax(eigenvalues)

        q[i, :] = eigenvectors[:, max_idx].real
        q[i, :] = [q[i, 3], q[i, 0], q[i, 1], q[i, 2]]

    return q


def q_from_axis_fun(axis_x, axis_y, axis_z):
    axis_x = np.array(axis_x)
    axis_y = np.array(axis_y)
    axis_z = np.array(axis_z)

    if np.sum(axis_x) == 0:
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = axis_z / np.linalg.norm(axis_z)
        tmp_x = np.cross(axis_y, axis_z)
        axis_x = tmp_x / np.linalg.norm(tmp_x)
    elif np.sum(axis_y) == 0:
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_z = axis_z / np.linalg.norm(axis_z)
        tmp_y = np.cross(axis_z, axis_x)
        axis_y = tmp_y / np.linalg.norm(tmp_y)
    elif np.sum(axis_z) == 0:
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        tmp_z = np.cross(axis_x, axis_y)
        axis_z = tmp_z / np.linalg.norm(tmp_z)
    else:
        axis_x = axis_x / np.linalg.norm(axis_x)
        axis_y = axis_y / np.linalg.norm(axis_y)
        axis_z = axis_z / np.linalg.norm(axis_z)
        if np.dot(axis_x, axis_y) + np.dot(axis_x, axis_z) + np.dot(axis_y, axis_z) != 0:
            raise ValueError('Wrong Axis Input, Three Axis are not Orthogonal!')

    dcm = np.array([axis_x, axis_y, axis_z])
    q_standard_2_rot = rotMat2quatern(dcm)
    return q_standard_2_rot

from ahrs.common.orientation import q_conj, q_prod

import numpy as np


def quat_mean_fun(q):
    q_len = q.shape[0]

    # Normalize quaternions
    for i in range(q_len):
        q[i, :] /= np.linalg.norm(q[i, :])

    # Apply sign correction
    sign_correction = np.diag((q[:, 0] > 0) * 2 - 1)
    q = np.matmul(sign_correction, q)

    cos_th = q[:, 0]
    sin_th = np.sqrt(1.0 - np.square(cos_th))

    v = q[:, 1:4]
    v[:, 0] /= sin_th
    v[:, 1] /= sin_th
    v[:, 2] /= sin_th

    # Compute mean of v
    v_mean = np.mean(v, axis=0)
    v_mean /= np.linalg.norm(v_mean)

    theta_2 = np.arccos(cos_th)
    theta_mean_2 = np.mean(theta_2)

    q_mean = np.array([np.cos(theta_mean_2), v_mean[0] * np.sin(theta_mean_2),
                       v_mean[1] * np.sin(theta_mean_2), v_mean[2] * np.sin(theta_mean_2)])

    return q_mean


def _process_static_data(static_data_gyro1, static_data_acc1):
    orientation1 = Mahony(gyr=static_data_gyro1, acc=static_data_acc1)

    z_pelvisCalib_inG = qvq_fun(orientation1.Q, np.array([0, 0, 1]))
    z_pelvisCalib_inG_onHor = np.hstack((np.mean(z_pelvisCalib_inG[:, :2], axis=0), 0))
    x_pelvisH0_inG = z_pelvisCalib_inG_onHor / np.linalg.norm(z_pelvisCalib_inG_onHor)
    y_pelvisH0_inG = [0, 0, 1]
    q_G2H_pel = q_from_axis_fun(x_pelvisH0_inG, y_pelvisH0_inG, [0, 0, 0])
    q_G2S_prox_mean = quat_mean_fun(orientation1.Q)
    q_S2H_prox_calib = quatProduct(q_conj(q_G2S_prox_mean).reshape(-1, 4), q_G2H_pel.reshape(-1, 4))
    return q_S2H_prox_calib


def _calibrate_raw_data(path):
    # 取前1s数据做标定,每一个trial对应一个标定，目前假设这个path是一个trial
    columns = [x + y for y in ['1', '2', '3'] for x in ['AccelX_', 'AccelY_', 'AccelZ_', 'GyroX_', 'GyroY_', 'GyroZ_']]
    trial_data = pd.read_excel(path, sheet_name='Sheet1', usecols=columns)
    static_data = trial_data.iloc[:100, :]
    static_data_acc1 = np.array(static_data[['AccelX_1', 'AccelY_1', 'AccelZ_1']])
    static_data_gyro1 = np.radians(np.array(static_data[['GyroX_1', 'GyroY_1', 'GyroZ_1']]))
    static_data_gyro_bias1 = np.mean(static_data_gyro1, axis=0)
    static_data_gyro1 -= static_data_gyro_bias1
    q_S2H_prox_calib1 = _process_static_data(static_data_gyro1, static_data_acc1)

    static_data_acc2 = np.array(static_data[['AccelX_2', 'AccelY_2', 'AccelZ_2']])
    static_data_gyro2 = np.radians(np.array(static_data[['GyroX_2', 'GyroY_2', 'GyroZ_2']]))
    static_data_gyro_bias2 = np.mean(static_data_gyro2, axis=0)
    static_data_gyro2 -= static_data_gyro_bias2
    q_S2H_prox_calib2 = _process_static_data(static_data_gyro2, static_data_acc2)

    static_data_acc3 = np.array(static_data[['AccelX_3', 'AccelY_3', 'AccelZ_3']])
    static_data_gyro3 = np.radians(np.array(static_data[['GyroX_3', 'GyroY_3', 'GyroZ_3']]))
    static_data_gyro_bias3 = np.mean(static_data_gyro3, axis=0)
    static_data_gyro3 -= static_data_gyro_bias3
    q_S2H_prox_calib3 = _process_static_data(static_data_gyro3, static_data_acc3)

    return q_S2H_prox_calib1, q_S2H_prox_calib2, q_S2H_prox_calib3



