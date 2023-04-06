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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms
import matplotlib.pyplot as plt



"---------------------Set Configurations----------------------------"
# Hyperparameters
lr = 0.1
epoch = 20
batch_size = 64
#regularization
#LAMBDA = 0.001

"---------------------Set Configurations----------------------------"
loss_list = []
acc_train_list = []
acc_val_list = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 设置随机数种子
random.seed(42)

# Set Data Path
DATA_DIR = Path().cwd() / 'cassava-leaf-disease-classification'
TRAIN_DIR = Path().cwd() / 'cassava-leaf-disease-classification/amls_train'
TEST_DIR = Path().cwd() / 'cassava-leaf-disease-classification/amls_test'
VAL_DIR = Path().cwd() / 'cassava-leaf-disease-classification/amls_valid'
model_path = Path().cwd() / 'cassava-leaf-disease-classification/model'

def train_test_val_split(DATA_DIR, TRAIN_DIR, TEST_DIR, VAL_DIR,model_path):
    # make directions
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # 读取CSV文件并打乱顺序
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    df = df.sample(frac=1, random_state=42)

    # 划分训练集和测试集
    train_ratio = 0.8
    test_ratio = 0.1
    val_ratio = 0.1

    train_idx = int(train_ratio * len(df))
    test_idx = train_idx + int(test_ratio * len(df))
    train_df = df[:train_idx]
    test_df = df[train_idx:test_idx]
    val_df = df[test_idx:]

    # 将CSV文件分别存储到训练集和测试集文件夹中
    train_df.to_csv(os.path.join(TRAIN_DIR, 'amls_train.csv'), index=False)
    test_df.to_csv(os.path.join(TEST_DIR, 'amls_test.csv'), index=False)
    val_df.to_csv(os.path.join(VAL_DIR, 'amls_valid.csv'), index=False)

    # 复制训练集图片到训练集文件夹中
    for image_name in train_df['image_id']:
        src_path = os.path.join(DATA_DIR, 'train_images', f'{image_name}')
        dst_path = os.path.join(TRAIN_DIR, f'{image_name}')
        shutil.copy(src_path, dst_path)

    # 复制测试集图片到测试集文件夹中
    for image_name in test_df['image_id']:
        src_path = os.path.join(DATA_DIR, 'train_images', f'{image_name}')
        dst_path = os.path.join(TEST_DIR, f'{image_name}')
        shutil.copy(src_path, dst_path)

    # 复制测试集图片到测试集文件夹中
    for image_name in val_df['image_id']:
        src_path = os.path.join(DATA_DIR, 'train_images', f'{image_name}')
        dst_path = os.path.join(VAL_DIR, f'{image_name}')
        shutil.copy(src_path, dst_path)

trans_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trans_valid = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


log_path = pathlib.Path.cwd() / ("VGG19_train_validation_" + "lr_"+ str(lr)+ "_"+str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".log")
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=log_path,
                    filemode='a')

# Load Dataset
class amls2Dataset(Dataset):
    def __init__(self, path, mode):
        self.all_image_paths = list(path.glob('*.jpg'))
        # print(len(self.all_image_paths))
        # load labels
        self.mode = mode
        if mode == "training":
            label_path = path / 'amls_train.csv'
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
        if self.mode == 'training':
            img = trans_train(img)
        else:
            img = trans_valid(img)
        label = self.all_image_labels[index]
        label = torch.tensor(label)
        return img, label

    # 看总共有多少张图片，是pytorch函数内部调用
    def __len__(self):
        return len(self.all_image_paths)


def create_csv(path, result_list):  # 不重要
    # save predict labels of test dataset
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(["predict_label", "gt_label", "match"])
        csv_write.writerows(result_list)

def record_acc_csv(filename, list):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in list:
            writer.writerow([item])

def train(net, train_iter, val_iter, criterion, optimizer, num_epochs):
    net = net.to(device)
    logging.info("-----training on %s-----", str(device))
    print("-----training on ", str(device), "-----")
    print(net)

    whole_batch_count = 0
    # training loop
    for epoch in range(num_epochs):
        start = time.time()
        net.train()  # trainning mode
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        # 放进cuda
        for X, y in train_iter:  # X: 图片 y: label
            X, y = X.to(device), y.to(device)
            y = y.to(torch.long)  # long=一个数据类型

            optimizer.zero_grad()  # 把optimizer的梯度设成0，清零
            y_hat = net(X)
            # print(y_hat.type(),y.type())
            loss = criterion(y_hat, y)  # loss function
            loss.backward()  # 调整参数，梯度传到参数上
            optimizer.step()  # 对参数进行调整从step上开始

            # 算一下我的loss、，算训练集准确率；算+打印   *不重要
            n += y.shape[0]
            whole_batch_count += 1
            batch_count += 1
            train_loss_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            temp_loss = train_loss_sum / whole_batch_count
            temp_acc_train = train_acc_sum / n
            loss_list.append(loss.item())
            # acc_train_list.append(temp_acc_train)
            logging.info('-epoch %d, batch_count %d, img nums %d, loss temp %.4f, train acc temp %.3f, time %.1f sec,'
                         % (epoch + 1, whole_batch_count, n, loss.item(), temp_acc_train, time.time() - start))
            print('-epoch %d, batch_count %d, img nums %d, loss temp %.4f, train acc temp %.3f, time %.1f sec'
                  % (epoch + 1, whole_batch_count, n, loss.item(), temp_acc_train, time.time() - start))

        # 在测试集上操作
        # test dataset inference will be done after each epoch
        with torch.no_grad():
            net.eval()  # evaluate mode
            val_acc_sum, n2 = 0.0, 0  # 初始化
            val_result_list = []
            for X, y in val_iter:
                y_hat = net(X.to(device))
                val_acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                temp = torch.stack((y_hat.argmax(dim=1).int(), y.to(device).int(), y_hat.argmax(dim=1) == y.to(device)),
                                   1).tolist()
                val_result_list.extend(temp)
                n2 += y.shape[0]
        # 计算+输出 不重要
        temp_acc_val = val_acc_sum / n2
        acc_val_list.append(temp_acc_val)
        acc_train_list.append(train_acc_sum / n)
        logging.info('---epoch %d, loss %.4f, train acc %.3f, validation acc %.3f, time %.1f sec---'
                     % (epoch + 1, temp_loss, train_acc_sum / n, temp_acc_val, time.time() - start))
        print('---epoch %d, loss %.4f, train acc %.3f,  validation acc %.3f, time %.1f sec---'
              % (epoch + 1, temp_loss, train_acc_sum / n, temp_acc_val, time.time() - start))
        # 存csv和路径
        result_path = Path().cwd() / ("epoch_" + str(epoch) + "_lr_" + str(lr) + "_valid_result.csv")
        create_csv(result_path, val_result_list)
        # save model
        torch.save(net.state_dict(),
           model_path / (str(type(net).__name__)+ "_epoch_" + str(epoch) + "_lr_" + str(lr) + "_" + str(
               time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".pth"))


# 计算每次epoch的match的总数除以epoch中test的次数，得到test accuracy；比对出最高的accuracy
def plot_save(loss_list, acc_list, train_acc_list):  # 不重要
    # plot temporary loss of training and accuracy of test dataset after each epoch training
    x1 = range(1, len(acc_list) + 1)  # 将起始值设置为 1
    x2 = range(len(loss_list))
    x3 = range(1, len(train_acc_list) + 1)
    y1 = acc_list
    y2 = loss_list
    y3 = train_acc_list

    # 创建画布和子图
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    # 上面那张图的标题是test accuracy vs. epoch no.，xlabel是epochs, ylabel是test accuracy
    axs[0].plot(x1, y1, 'o-')
    axs[0].set_title('Validation Accuracy vs. Epoch No.')
    axs[0].set_xlabel('Epochs No.')
    axs[0].set_ylabel('Validation Accuracy')

    axs[1].plot(x3, y3, 'o-')
    axs[1].set_title('Training Accuracy vs. Epoch No.')
    axs[1].set_xlabel('Epochs No.')
    axs[1].set_ylabel('Training dataset Accuracy')
    # 下面那张图的标题是training loss vs. batch count, xlabel是batch count no.，ylabel是training loss
    axs[2].plot(x2, y2, '.-')
    axs[2].set_title('Training Loss vs. Batch Count')
    axs[2].set_xlabel('Batch Count No.')
    axs[2].set_ylabel('Training Loss')

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(("VGG_lr_" + str(lr) + "_" + str(
        time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".jpg"))
    plt.clf()  # Clear figure
    plt.cla()  # Clear axes
    plt.close()


#VGG
# 获取vgg19原始模型, 输出图像维度是1000.
# 下面三行代码功能是:得到修改后的vgg19模型.
# 具体实现是: 去掉vgg19原始模型的第三部分classifier的最后一个全连接层,
# 用新的分类器替换原始vgg19的分类器，使输出维度是4096.
#train_test_val_split(DATA_DIR, TRAIN_DIR, TEST_DIR, VAL_DIR, model_path)
train_dataset = amls2Dataset(TRAIN_DIR, "training")
val_dataset = amls2Dataset(VAL_DIR, "validation")

train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_iter = DataLoader(val_dataset, batch_size=batch_size)

pretrained_net = models.vgg19(pretrained=True)
num_classes = 5  # 设置新的分类数
pretrained_net.classifier[-1] = nn.Linear(4096, num_classes)
# 把最后一层全连接层的parameters分离出来
optimizer = optim.SGD([{'params': pretrained_net.features.parameters()},
                       {'params': pretrained_net.avgpool.parameters()},
                       {'params': pretrained_net.classifier.parameters(), 'lr': lr * 10}],  # fc的学习率设高一点能学习的更好
                      lr=lr, weight_decay=0.001)

loss = torch.nn.CrossEntropyLoss()
train(pretrained_net, train_iter, val_iter, loss, optimizer, num_epochs=epoch)
plot_save(loss_list, acc_val_list, acc_train_list)
torch.save(pretrained_net.state_dict(),
           model_path / ("epoch_" + str(epoch) + "_lr_" + str(lr) + "_" + str(
               time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".pth"))

record_acc_csv('vgg_test_acc.csv', acc_val_list)