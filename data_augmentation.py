# -*- coding: utf-8 -*-
# @Time : 2023/5/12 10:26
# @Author : Wanghong
# @FileName: data_augmentation.py
# @Software: PyCharm
# Reference: T. T. Um, F. M. Pfister, D. Pichler, S. Endo, M. Lang, S. Hirche,
# U. Fietzek, and D. Kulic, “Data augmentation of wearable sensor data ´
# for parkinson’s disease monitoring using convolutional neural networks,”
# Proceedings of the 19th ACM international conference on multimodal
# interaction, pp. 216–220, 2017
import numpy as np
import torch
import random
import matplotlib.pyplot as plt


def random_time_shift(data, max_shift):
    # Generate a random shift value within the specified range
    shift = random.randint(-max_shift, max_shift)

    if shift == 0:
        # If the shift value is 0, return the original data
        return data

    elif shift > 0:
        # If the shift value is positive, shift the data forward
        shifted_data = np.vstack((data[shift:, :], data[:shift, :]))

    else:
        # If the shift value is negative, shift the data backward
        shifted_data = np.vstack((data[shift:, :], data[:shift, :]))

    return shifted_data


def add_gauss_noise(data, sigma=0.5, mu=0):
    noised_data = data + sigma * np.random.randn(*data.shape)
    return noised_data


# ## 1. Jittering
def DA_Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise


# ## 2. Scaling
def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise


from scipy.interpolate import CubicSpline
## This example using cubic splice is not the best approach to generate random curves.
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    new_X = np.zeros(X.shape)
    for i in range(X.shape[1]):
        new_X[:, i] = CubicSpline(xx[:, i], yy[:, i])(x_range)
    return new_X


# ## 3. Magnitude Warping
# "Scaling" can be considered as "applying constant noise to the entire samples" whereas "Jittering" can be considered as "applying different noise to each sample".
# "Magnitude Warping" can be considered as "applying smoothly-varing noise to the entire samples"
def DA_MagWarp(X, sigma=0.2):
    return X * GenerateRandomCurves(X, sigma)


def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = []
    for i in range(X.shape[1]):
        t_scale.append((X.shape[0]-1)/tt_cum[-1, i])
    for i in range(X.shape[1]):
        tt_cum[:, i] = tt_cum[:, i] * t_scale[i]
    return tt_cum


# ## 4. Time Warping
# todo: 特征重新提取
def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    for i in range(18):
        X_new[:, i] = np.interp(x_range, tt_new[:, i], X[:, i])

    X_new[:, 18] = np.linalg.norm(X_new[:, 0:3], axis=1)
    X_new[:, 19] = np.linalg.norm(X_new[:, 3:6], axis=1)
    X_new[:, 20] = np.mean(X_new[:, 0:3], axis=1)
    X_new[:, 21] = np.mean(X_new[:, 3:6], axis=1)
    X_new[:, 22] = np.linalg.norm(X_new[:, 6:9], axis=1)
    X_new[:, 23] = np.linalg.norm(X_new[:, 9:12], axis=1)
    X_new[:, 24] = np.mean(X_new[:, 6:9], axis=1)
    X_new[:, 25] = np.mean(X_new[:, 9:12], axis=1)
    X_new[:, 26] = np.linalg.norm(X_new[:, 12:15], axis=1)
    X_new[:, 27] = np.linalg.norm(X_new[:, 15:18], axis=1)
    X_new[:, 28] = np.mean(X_new[:, 12:15], axis=1)
    X_new[:, 29] = np.mean(X_new[:, 15:18], axis=1)
    return X_new


# ## 6. Permutation
def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)


# ## 5. Rotation
def DA_Rotation(X):
    from transforms3d.axangles import axangle2mat
    axis = np.random.uniform(low=-1, high=1, size=3)
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X, axangle2mat(axis,angle))


# ## 6. Crop
def DA_Crop(X, len, crop=10):
    # remove 1/10 data from start or end
    length = len
    length_one_tenth = int(length/10)
    length_nine_tenth = length - length_one_tenth

    start = np.random.randint(0, length_one_tenth)
    end = np.random.randint(length_nine_tenth, length)

    return np.vstack((X[:start,:], X[end:,:]))

