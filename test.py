# -*- coding: utf-8 -*-
"""
@Time ： 2024/7/3 16:28
@Auth ： Wang Hong
@File ：test.py
"""
import torch

# 查看 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 查看 CUDA 版本
print(f"是否支持 CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"CUDA 驱动版本: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")


import numpy as np
import h5py
import sklearn

# 查看 NumPy 版本
print(f"NumPy 版本: {np.__version__}")

# 查看 h5py 版本
print(f"h5py 版本: {h5py.__version__}")

# 查看 Scikit-learn 版本
print(f"Scikit-learn 版本: {sklearn.__version__}")
