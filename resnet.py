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
from resnet_customized import resnet18_customized


"---------------------Set Configurations----------------------------"
# Hyperparameters
lr = 0.001
epoch = 20
batch_size = 64

loss_list = []
acc_train_list = []
acc_val_list = []

# check if the program is running on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# set random seeds
random.seed(42)

# Set Data Path
DATA_DIR = Path().cwd() / 'Datasets/cassava-leaf-disease-classification'
TRAIN_DIR = Path().cwd() / 'Datasets/cassava-leaf-disease-classification/amls_train'
TEST_DIR = Path().cwd() / 'Datasets/cassava-leaf-disease-classification/amls_test'
VAL_DIR = Path().cwd() / 'Datasets/cassava-leaf-disease-classification/amls_valid'
model_path = Path().cwd() / 'Datasets/cassava-leaf-disease-classification/model'
"---------------------Set Configurations----------------------------"


def train_test_val_split(DATA_DIR, TRAIN_DIR, TEST_DIR, VAL_DIR, model_path):
    '''
    Split train datasets into train/validation/test datasets; model(.pth files) saved path
    Args:
        DATA_DIR: Data Directory to store any related files towards this task, including train, validation and
                    test images with related labelled csv file, model save path, etc.
        TRAIN_DIR: Training dataset directory
        TEST_DIR: Testing dataset directory
        VAL_DIR: Validation dataset directory
        model_path: Model (.pth file) saved directory

    Returns: No returns

    '''
    # make directions
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # read csv file and shuffle
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    df = df.sample(frac=1, random_state=42)

    # set dataset splitting ratios
    train_ratio = 0.8
    test_ratio = 0.1
    val_ratio = 0.1

    train_idx = int(train_ratio * len(df))
    test_idx = train_idx + int(test_ratio * len(df))
    train_df = df[:train_idx]
    test_df = df[train_idx:test_idx]
    val_df = df[test_idx:]

    # save the CSV file separately into training, validation and testing , and set directories.
    train_df.to_csv(os.path.join(TRAIN_DIR, 'amls_train.csv'), index=False)
    test_df.to_csv(os.path.join(TEST_DIR, 'amls_test.csv'), index=False)
    val_df.to_csv(os.path.join(VAL_DIR, 'amls_valid.csv'), index=False)

    # Copy the training set image to the training dataset directory
    for image_name in train_df['image_id']:
        src_path = os.path.join(DATA_DIR, 'train_images', f'{image_name}')
        dst_path = os.path.join(TRAIN_DIR, f'{image_name}')
        shutil.copy(src_path, dst_path)

    # Copy the testing set image to the testing dataset directory
    for image_name in test_df['image_id']:
        src_path = os.path.join(DATA_DIR, 'train_images', f'{image_name}')
        dst_path = os.path.join(TEST_DIR, f'{image_name}')
        shutil.copy(src_path, dst_path)

    # Copy the validation set image to the validation dataset directory
    for image_name in val_df['image_id']:
        src_path = os.path.join(DATA_DIR, 'train_images', f'{image_name}')
        dst_path = os.path.join(VAL_DIR, f'{image_name}')
        shutil.copy(src_path, dst_path)

''' 
Data Augmentation: 
two sets of data augmentations given, one for training images, one for validation images
'''

trans_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trans_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

'''
logging path setting:
Setting format for data logging to be saved on the local disk
'''
log_path = pathlib.Path.cwd() / ("Resnet18_train_validation_" + "lr_"+ str(lr) + "_" + str(time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".log")
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=log_path,
                    filemode='a')


class amls2Dataset(Dataset):
    # make image path corresponds to its label, and check whether all data are loaded and matched
    def __init__(self, path, mode):
        '''

        Args:
            path: directory will be sent to the model
            mode: training/validation
        '''
        self.all_image_paths = list(path.glob('*.jpg'))
        self.mode = mode
        if mode == "training":
            label_path = path / 'amls_train.csv'
        else:
            label_path = path/ 'amls_valid.csv'
        label_list = self.load_label(label_path, mode)
        #print(label_list)
        label_dict = dict((temp[0], temp[1]) for temp in label_list)
        #print(len(label_dict))
        # Check label amount with image amount, whether matches; helps uploading datasets
        if len(label_dict) != len(self.all_image_paths):
            logging.warning('-----label amount not match with img amount-----')
            print('-----label amount not match with img amount-----')

        # corresponds the label to image
        self.all_image_labels = list()
        for i in self.all_image_paths:
            if label_dict.get(str(i.name)) is not None:
                self.all_image_labels.append(float(label_dict[str(i.name)]))
            else:
                logging.warning('-----no label imgs-----')
                print('-----no label imgs-----')
                print(i)

    def load_label(self, path, mode):
        '''
        loading labels from csv file to list
        Args:
            path: directory to access(train/validation)
            mode: validation/training

        Returns:
            label_data: a list where image_id matches labels, images are of the inputted directory

        '''
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

    def __getitem__(self, index):
        '''
        works for data augmentation
        Args:
            index: label indexes

        Returns:
            img: corresponded images
            label: corresponded labels
        '''
        img = Image.open(self.all_image_paths[index])
        if self.mode == 'training':
            img = trans_train(img)
        else:
            img = trans_valid(img)
        label = self.all_image_labels[index]
        label = torch.tensor(label)
        return img, label

    def __len__(self):
        '''

        Returns: size of the dataset

        '''
        return len(self.all_image_paths)


def create_csv(path, result_list):
    '''
    used in train(), to record the training/validation results derived into csv file for further debugging
    Args:
        path: path to store results
        result_list: list stored

    Returns:

    '''
    with open(path, 'w', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(["predict_label", "gt_label", "match"])
        csv_write.writerows(result_list)

def record_acc_csv(filename, list):
    '''
    record the accuracy for 20 epochs, store locally
    Args:
        filename: file name
        list: accuracy list for 20 epochs

    Returns: None

    '''
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in list:
            writer.writerow([item])

def train(net, train_iter, val_iter, criterion, optimizer, num_epochs):
    '''
    train the network
    Args:
        net: utilized model
        train_iter: Training dataloader to group data into batches and set shuffle
        val_iter: Validation dataloader to group data into batches and set shuffle
        criterion: Calculate loss
        optimizer: Optimize algorithms parameters
        num_epochs: setting epochs to train

    Returns:

    '''
    net = net.to(device)
    logging.info("-----training on %s-----", str(device))
    print("-----training on ", str(device), "-----")
    print(net)

    whole_batch_count = 0
    # start training loop
    for epoch in range(num_epochs):
        start = time.time()
        net.train()  # trainning mode
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for X, y in train_iter:  # X: images y: labels
            X, y = X.to(device), y.to(device)
            y = y.to(torch.long)

            optimizer.zero_grad()
            y_hat = net(X)
            loss = criterion(y_hat, y)  # loss function
            loss.backward()  # compute the gradients of the loss function, w.r.t. he parameters of a model
            optimizer.step()  # update the parameters of a model based on the gradients computed using backpropagation

            # Calculate loss, training dataset accuracy
            n += y.shape[0]
            whole_batch_count += 1
            batch_count += 1
            train_loss_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            temp_loss = train_loss_sum / whole_batch_count
            temp_acc_train = train_acc_sum / n
            loss_list.append(loss.item())
            logging.info('-epoch %d, batch_count %d, img nums %d, loss %.4f, train acc %.3f, time %.1f sec,'
                         % (epoch + 1, whole_batch_count, n, loss.item(), temp_acc_train, time.time() - start))
            print('-epoch %d, batch_count %d, img nums %d, loss %.4f, train acc  %.3f, time %.1f sec'
                  % (epoch + 1, whole_batch_count, n, loss.item(), temp_acc_train, time.time() - start))

        # Model Inference on validation dataset, after each epoch
        with torch.no_grad():
            net.eval()  # evaluate mode
            val_acc_sum, n2 = 0.0, 0
            val_result_list = []
            for X, y in val_iter:
                y_hat = net(X.to(device))
                val_acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                temp = torch.stack((y_hat.argmax(dim=1).int(), y.to(device).int(), y_hat.argmax(dim=1) == y.to(device)),
                                   1).tolist()
                val_result_list.extend(temp)
                n2 += y.shape[0]
        # Calculate loss, validation dataset accuracy
        temp_acc_val = val_acc_sum / n2
        acc_val_list.append(temp_acc_val)
        acc_train_list.append(train_acc_sum / n)
        logging.info('---epoch %d, loss %.4f, train acc %.3f, validation acc %.3f, time %.1f sec---'
                     % (epoch + 1, temp_loss, train_acc_sum / n, temp_acc_val, time.time() - start))
        print('---epoch %d, loss %.4f, train acc %.3f,  validation acc %.3f, time %.1f sec---'
              % (epoch + 1, temp_loss, train_acc_sum / n, temp_acc_val, time.time() - start))
        # save result csv file
        result_path = Path().cwd() / ("epoch_" + str(epoch) + "_lr_" + str(lr) + "_valid_result.csv")
        create_csv(result_path, val_result_list)
        # save model .pth file
        torch.save(net.state_dict(),
           model_path / (str(type(net).__name__)+ "_epoch_" + str(epoch) + "_lr_" + str(lr) + "_" + str(
               time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".pth"))


def plot_save(loss_list, acc_list, train_acc_list):  # 不重要
    '''
    Three figures, validation accuracy vs. epoch no.; training accuracy vs. epoch no.; training loss vs. Batch count
    Args:
        loss_list: training loss for all batches
        acc_list: validation accuracy for 20 epochs
        train_acc_list: training accuracy for 20 epochs

    Returns: three figures

    '''
    x1 = range(1, len(acc_list) + 1)  # set start value to 1
    x2 = range(len(loss_list))
    x3 = range(1, len(train_acc_list) + 1)
    y1 = acc_list
    y2 = loss_list
    y3 = train_acc_list

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(x1, y1, 'o-')
    axs[0].set_title('Validation Accuracy vs. Epoch No.')
    axs[0].set_xlabel('Epochs No.')
    axs[0].set_ylabel('Validation Accuracy')

    axs[1].plot(x3, y3, 'o-')
    axs[1].set_title('Training Accuracy vs. Epoch No.')
    axs[1].set_xlabel('Epochs No.')
    axs[1].set_ylabel('Training dataset Accuracy')

    axs[2].plot(x2, y2, '.-')
    axs[2].set_title('Training Loss vs. Batch Count')
    axs[2].set_xlabel('Batch Count No.')
    axs[2].set_ylabel('Training Loss')

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(("Resnet_lr_" + str(lr) + "_" + str(
        time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".jpg"))
    plt.clf()  # Clear figure
    plt.cla()  # Clear axes
    plt.close()

def run():
    train_test_val_split(DATA_DIR, TRAIN_DIR, TEST_DIR, VAL_DIR, model_path)
    train_dataset = amls2Dataset(TRAIN_DIR, "training")
    val_dataset = amls2Dataset(VAL_DIR, "validation")

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_iter = DataLoader(val_dataset, batch_size=batch_size)

    # Define the resnet model, two options: pretrained resnet-18, customized resnet-18
    # the pretrained model has the fc layer for 1000 classes, which need to be modified to 5
    # --------------If want to use pretrained resnet-18, not comment these three lines, comment the line of 395 -------
    pretrained_net = models.resnet18(pretrained=True)
    print("-----------------------Running:",  type(pretrained_net).__name__, "---------------------------------------")
    num_ftrs = pretrained_net.fc.in_features
    pretrained_net.fc = nn.Linear(num_ftrs, 5)

    # --------------If using customized ResNet-18 , using this line of code, commented last 3 lines above-------------
    # pretrained_net = resnet18_without_residuals(5, pretrained=False)

    #  Separating the parameters of the last fully connected layer, as fc layer performs better in greater learning rate
    output_params = list(map(id, pretrained_net.fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

    # Apply optimizer
    optimizer = optim.SGD([{'params': feature_params},
                           {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                          lr=lr, weight_decay=0.001, momentum=0.9)
    # Calculate Loss
    loss = torch.nn.CrossEntropyLoss()
    print("The Nueral Network is: ", type(pretrained_net).__name__)
    train(pretrained_net, train_iter, val_iter, loss, optimizer, num_epochs=epoch)
    plot_save(loss_list, acc_val_list, acc_train_list)
    # save model
    torch.save(pretrained_net.state_dict(),
               model_path / ("epoch_" + str(epoch) + "_lr_" + str(lr) + "_" + str(
                   time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".pth"))

    record_acc_csv('resnet_test_acc.csv', acc_val_list)

if __name__ == '__main__':
    run()


