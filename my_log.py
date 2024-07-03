import csv
import time
import os
import numpy as np
from matplotlib import pyplot as plt
# import const


class MyLog:
    """
    要记录：模型名称，模型参数，优化器名称，优化器参数，epoch数，学习率以及学习率策略，模型精度，训练集验证集和测试集loss
    "模型 : {}, 模型参数 : {}, 优化器 : {}, 优化器参数 : {}, epoch : {}, 学习率 : {}, 传感器 : {}".format()
    """

    def __init__(self, path, folder_name, file_type="txt"):
        self.log_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        self.file_name = folder_name
        self.dic_path = path + '/' + folder_name + self.log_time
        self.log_path = path + '/' + folder_name + self.log_time + '/' + 'log.' + file_type
        self.message_list = []
        self.image_dic = {}
        self.csv_list = []
        self.add_message("Log Time : " + self.log_time)

    def add_message(self, message):
        self.message_list.append(message)

    def add_image(self, image, image_name, image_type="png"):
        image_path = self.dic_path + '/' + image_name + '.' + image_type
        self.image_dic[image_path] = image

    def add_csv(self, csv_data, csv_name, data_type=".csv"):
        self.csv_path = self.dic_path + '/' + csv_name + data_type
        self.csv_list.append(csv_data)

    def save(self):
        if not os.path.exists(self.dic_path):
            os.makedirs(self.dic_path)
        if not self.message_list:
            print("message_list is empty")
        else:
            with open(self.log_path, "a") as f:
                for message in self.message_list:
                    f.write(message + '\n')
            self.message_list = []
        if not self.image_dic:
            pass
            # print("image_dic is empty")
        else:
            for image_path in self.image_dic.keys():
                plt.imsave(image_path, self.image_dic[image_path])
            self.image_dic.clear()
        if not self.csv_list:
            pass
            # print("image_dic is empty")
        else:
            with open(self.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                for csv_data in self.csv_list:
                    writer.writerows(csv_data)
            self.csv_list = []


