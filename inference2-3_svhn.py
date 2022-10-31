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
import pandas as pd

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


class Digits_test(Dataset):
    def __init__(self, data_root, domain, mode, transform=None):
        self.root = data_root
        self.domain = domain
        self.mode = mode
        self.transform = transform

        # # First, read data_list.
        # with open(csv_name, 'r')as f:
        #     next(f)
        #     data_list = f.readlines()

        self.img_paths = os.listdir(data_root)
        self.n_data = len(self.img_paths)

        # Then, read image from data_list.
        # for data in data_list:
        #     self.img_paths.append(data[:-3]) # "00002.png",4<eol>
        #     self.img_labels.append(data[-2]) # 00002.png,"4"<eol>
        #print(len(data_list))

    def __getitem__(self, item):
        img_name = self.img_paths[item]
        img_path = os.path.join(self.root, img_name)
        imgs = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, img_name

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
    alpha = 1
    result = []
    img_names = []
    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, img_name = data_target
        #print(t_img.shape)

        t_img = t_img.to(device)

        class_output, _ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        for p in pred:
            result.append(p.cpu().numpy()[0])
        #print(img_name)
        for name in img_name:
            #print(name)
            img_names.append(name)
        i += 1
        print(i, end = '\r')

    return result, img_names


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="hw 2-3 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data_dir", help="Training data location")
    parser.add_argument("output_dir", help="Output data location")
    
    parser.add_argument("--ckpt_dir", help="Checkpoint location", default="ckpt2-3_svhn")
    parser.add_argument("--batch_size", help="batch size", type=int, default=128)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=2e-4)
    parser.add_argument("--n_epoch", help="n_epoch", type=int, default=1000)
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
    os.makedirs(ckpt_dir, exist_ok=True)

    batch_size = args.batch_size
    # lr = args.learning_rate
    n_epoch = args.n_epoch
    image_size = 28

    

    img_transform_target = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # svhn is RGB
    ])

    # Test loader
    # /hw2_data/digits/svhn/test
    # /hw2_data/digits/svhn/data, split dataset by csv file
    val_dataset_target = Digits_test(  
        data_root=os.path.join(data_dir),
        domain = "target",
        mode = "val",
        transform=img_transform_target
    )
     # Test dataloader
    val_dataloader_target = DataLoader(
        dataset=val_dataset_target,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )   
    
    # load model
    my_net = CNNModel()
    ckpt_name = os.path.join(ckpt_dir, "best_model_41.pth")
    my_net.load_state_dict(torch.load(ckpt_name))
    my_net = my_net.to(device)
    my_net.eval()

    # testing
    print("Evaluation")
    my_net = my_net.eval()

    # Target domain
    result, img_names = test(val_dataloader_target, my_net)
    print(len(result))
    print(len(img_names))

    df = pd.DataFrame() # apply pd.DataFrame format 
    df["image_name"] = img_names
    df["label"] = result
    print(df)

    csv_name = output_dir.split('/')[-1]
    print(csv_name)
    dest_folder = output_dir.replace(csv_name,'') 
    #print(dest_folder)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    df.to_csv(output_dir, index = False)

