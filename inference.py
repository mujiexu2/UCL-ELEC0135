import csv
import numpy as np
import logging
import pathlib
import os
import random
import shutil
import time
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Recommended normalization params
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
batch_size = 64
acc_test_list = []

DATA_DIR = Path().cwd() / 'cassava-leaf-disease-classification'
TRAIN_DIR = Path().cwd() / 'cassava-leaf-disease-classification/amls_train'
TEST_DIR = Path().cwd() / 'cassava-leaf-disease-classification/amls_test'
VAL_DIR = Path().cwd() / 'cassava-leaf-disease-classification/amls_valid'

model_path_resnet = Path().cwd() /'cassava-leaf-disease-classification/model/ResNet_epoch_16_lr_0.001_04_02_05_23_52.pth'
model_path_vgg = Path().cwd() /'cassava-leaf-disease-classification/model/VGG_epoch_16_lr_0.001_04_04_01_54_28.pth'

trans_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trans_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class amls2Dataset(Dataset):
    def __init__(self, path, mode):
        self.all_image_paths = list(path.glob('*.jpg'))
        # print(len(self.all_image_paths))
        # load labels
        self.mode = mode
        if mode == "test":
            label_path = path / 'amls_test.csv'
        else:
            label_path = path/ 'amls_valid.csv'
        label_list = self.load_label(label_path, mode)
        print(label_list)
        label_dict = dict((temp[0], temp[1]) for temp in label_list)
        print(len(label_dict))
        # ground truth amount check
        if len(label_dict) != len(self.all_image_paths):
            logging.warning('-----label amount dismatch with img amount-----')
            print('-----label amount dismatch with img amount-----')

        # corresponding label to img
        self.all_image_labels = list()
        for i in self.all_image_paths:
            if label_dict.get(str(i.name)) is not None:
                self.all_image_labels.append(float(label_dict[str(i.name)]))
            else:
                logging.warning('-----no label imgs-----')
                print('-----no label imgs-----')
                print(i)

        # image normalization params
        # self.mean = np.array(mean).reshape((1, 1, 3))
        # self.std = np.array(std).reshape((1, 1, 3))
        # load label生成一个list，存所有按照顺序排列的图片的名字，label和index对应
    def load_label(self, path, mode):
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = []
            for i, row in enumerate(reader):
                if i == 0:
                    dataset_title = row
                    continue
                rows.append(row)
            label_data = np.array(rows)
        logging.info('-----load %s dataset labels-----', mode)
        print('-----load ', mode, ' dataset labels-----')
        return label_data

    # 每次iteration要调用一次getitem（16）；负责送图片和label；送给模型
    def __getitem__(self, index):
        img = Image.open(self.all_image_paths[index])
        if self.mode == 'test':
            img = trans_test(img)
        else:
            img = trans_valid(img)
        label = self.all_image_labels[index]
        label = torch.tensor(label)
        return img, label

    # 看总共有多少张图片，是pytorch函数内部调用
    def __len__(self):
        return len(self.all_image_paths)

def create_csv(path, result_list):
    # save predict labels of test dataset
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(["predict_label", "gt_label", "match"])
        csv_write.writerows(result_list)

def heatmap(test_result_list):
  # 通过列表解析取出每个小list的第一个值
    y_true = [sublist[0] for sublist in test_result_list]
    y_pred = [sublist[1] for sublist in test_result_list]

    class_names = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    # 绘制混淆矩阵图

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2%',
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    # 设置轴标签
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')

    # 显示图像
    # # 绘制混淆矩阵热图
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, cmap='Blues')
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    plt.show()


def inference(net, test_iter):
    net = net.to(device)
    logging.info("-----training on %s-----", str(device))
    print("-----training on ", str(device), "-----")
    print(net)

    start = time.time()

    with torch.no_grad():
        net.eval()  # evaluate mode
        test_acc_sum, n2 = 0.0, 0  # 初始化
        test_result_list = []
        for X, y in test_iter:
            y_hat = net(X.to(device))
            test_acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            temp = torch.stack((y_hat.argmax(dim=1).int(), y.to(device).int(), y_hat.argmax(dim=1) == y.to(device)),
                               1).tolist()
            test_result_list.extend(temp)
            n2 += y.shape[0]
    # 计算+输出 不重要
    temp_acc_test = test_acc_sum / n2
    acc_test_list.append(temp_acc_test)
    logging.info('---test acc %.4f, time %.1f sec---'
                 % (temp_acc_test, time.time() - start))
    print('---test acc %.4f, time %.1f sec---'
          % (temp_acc_test, time.time() - start))
    # 存csv和路径
    result_path = model_path_resnet.parent / ("inference_" + str(model_path_resnet.stem) + "_result.csv")
    create_csv(result_path, test_result_list)

    # result_path = model_path_vgg.parent / ("inference_" + str(model_path_vgg.stem) + "_result.csv")
    # create_csv(result_path, test_result_list)
    heatmap(test_result_list)

test_dataset = amls2Dataset(TEST_DIR, "test")
# 初始化Dataloader
test_iter = DataLoader(test_dataset, batch_size=batch_size)

pretrained_net = models.resnet18(pretrained=True)
# pretrained_net = models.vgg19(pretrained=True)
# 定义新全连接层
num_ftrs = pretrained_net.fc.in_features
pretrained_net.fc = nn.Linear(num_ftrs, 5)
# num_classes = 5  # 设置新的分类数
# pretrained_net.classifier[-1] = nn.Linear(4096, num_classes)

pretrained_net.load_state_dict(torch.load(pathlib.Path(model_path_resnet)))
net = pretrained_net.to(device)

inference(pretrained_net, test_iter)
