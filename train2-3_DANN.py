# DANN ref: https://github.com/fungtion/DANN

import os
import glob
import argparse
import random
from datetime import datetime
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# seed setting
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Digits(Dataset):
    def __init__(self, data_root, csv_name, domain, mode, transform=None):
        self.root = data_root
        self.domain = domain
        self.mode = mode
        self.transform = transform

        # First, read data_list.
        with open(csv_name, 'r')as f:
            next(f)
            data_list = f.readlines()

        self.n_data = len(data_list)
        self.img_paths = []
        self.img_labels = []

        # Then, read image from data_list.
        for data in data_list:
            self.img_paths.append(data[:-3]) # "00002.png",4<eol>
            self.img_labels.append(data[-2]) # 00002.png,"4"<eol>
        #print(len(data_list))

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

        
class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


def test(dataloader, my_net):
    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)
    i = 0
    n_total = 0
    n_correct = 0
    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        t_img = t_img.to(device)
        t_label = t_label.to(device)

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1
    accu = n_correct.data.numpy() * 1.0 / n_total
    return accu


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="hw 2-1 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", help="Output data location")

    parser.add_argument("--data_dir", help="Training data location", default="./hw2_data/digits")
    parser.add_argument("--mode", help="train or test", default="train")   
    parser.add_argument("--ckpt_dir", help="Checkpoint location", default="ckpt2-3")
    parser.add_argument("--batch_size", help="batch size", type=int, default=128)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=3e-4)
    parser.add_argument("--n_epoch", help="n_epoch", type=int, default=150)
    args = parser.parse_args()
    print(vars(args))
    
    same_seeds(1211)

    if torch.cuda.is_available():
        if torch.cuda.device_count()==2:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using", device)

    output_dir = args.output_dir
    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir

    batch_size = args.batch_size
    lr = args.learning_rate
    n_epoch = args.n_epoch
    image_size = 28

    source_dataset_name = 'mnistm'
    target_dataset_name = 'usps' 
    source_image_root = os.path.join(args.data_dir, source_dataset_name)
    target_image_root = os.path.join(args.data_dir, target_dataset_name)
    # print(source_image_root)
    # print(target_image_root)

    img_transform_source = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  
    ])
    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    # Train dataset
    train_dataset_source = Digits(  
        data_root=os.path.join(source_image_root, 'data'),
        csv_name=os.path.join(source_image_root, 'train.csv'),
        domain = "source",
        mode = "train",
        transform=img_transform_source
    )
    train_dataset_target = Digits(  
        data_root=os.path.join(target_image_root, 'data'),
        csv_name=os.path.join(target_image_root, 'train.csv'),
        domain = "target",
        mode = "train",
        transform=img_transform_target
    )
    # Train dataloader
    train_dataloader_source = DataLoader(
        dataset=train_dataset_source,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    train_dataloader_target = DataLoader(
        dataset=train_dataset_target,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    # Test loader
    val_dataset_source = Digits(  
        data_root=os.path.join(source_image_root, 'data'),
        csv_name=os.path.join(source_image_root, 'val.csv'),
        domain = "source",        
        mode = "val",
        transform=img_transform_source
    )
    val_dataset_target = Digits(  
        data_root=os.path.join(target_image_root, 'data'),
        csv_name=os.path.join(target_image_root, 'val.csv'),
        domain = "target",
        mode = "val",
        transform=img_transform_target
    )
     # Test dataloader
    val_dataloader_source = DataLoader(
        dataset=val_dataset_source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )
    val_dataloader_target = DataLoader(
        dataset=val_dataset_target,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )   
    


    # load model
    my_net = CNNModel()
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    
    my_net = my_net.to(device)
    loss_class = loss_class.to(device)
    loss_domain = loss_domain.to(device)

    for p in my_net.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(my_net.parameters(), lr=lr)




    # training
    best_accu_t = 0.0
    for epoch in range(n_epoch):

        len_dataloader = min(len(train_dataloader_source), len(train_dataloader_target))
        data_source_iter = iter(train_dataloader_source)
        data_target_iter = iter(train_dataloader_target)

        pbar = tqdm(range(len_dataloader))
        pbar.set_description(f"Epoch {epoch+1}")
        for i in pbar:

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = data_source_iter.next()
            s_img, s_label = data_source

            my_net.zero_grad()
            batch_size = len(s_label)

            domain_label = torch.zeros(batch_size).long()

            s_img = s_img.to(device)
            s_label = s_label.to(device)
            domain_label = domain_label.to(device)


            class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data
            data_target = data_target_iter.next()
            t_img, _ = data_target

            batch_size = len(t_img)

            domain_label = torch.ones(batch_size).long()

            t_img = t_img.to(device)
            domain_label = domain_label.to(device)

            _, domain_output = my_net(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)
            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            pbar.set_postfix(
                err_s_label = err_s_label.data.cpu().numpy(), 
                err_s_domain = err_s_domain.data.cpu().numpy(),
                err_t_domain = err_t_domain.data.cpu().item(),
            )
        
        torch.save(my_net.state_dict(), os.path.join(ckpt_dir, f'model_{epoch}.pth'))
        print("Evaluation")
        my_net = my_net.eval()

        # Target domain
        accu_t = test(val_dataloader_target, my_net)
        print(f'Accuracy of the {target_dataset_name} dataset: {accu_t}')

        # Source domain
        accu_s = test(val_dataloader_source, my_net)
        print(f'Accuracy of the {source_dataset_name} dataset: {accu_s}')

        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(my_net.state_dict(), os.path.join(ckpt_dir, f'best_model_{epoch}.pth'))
            best_epoch = epoch
        


    print('============ Summary ============= \n')
    print('Accuracy of the %s dataset: %f' % ('mnist', best_accu_s))
    print('Accuracy of the %s dataset: %f' % ('mnist_m', best_accu_t))
    print(f'Best epoch: {best_epoch}')