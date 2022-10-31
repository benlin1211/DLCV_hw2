import torch
import torchvision
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import tqdm
import matplotlib.pyplot as plt
# from utils.inception_score import inception_score
import numpy as np
from subprocess import Popen
import subprocess
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")

class P1(Dataset):
    def __init__(self, root, transform=None):
        self.filenames = sorted(os.listdir(root))
        self.root = root
        self.transform = transform
        self.len = len(self.filenames)
    def __getitem__(self, index):
        imagePath = os.path.join(self.root,self.filenames[index])
        image = Image.open(imagePath)
        if self.transform is not None:
            image = self.transform(image)
        return image
    def __len__(self):
        return self.len

class P1Eval(Dataset):
    def __init__(self, model, nz, transform=None):
        self.imgs = []
        if device == "cpu":
            torch.manual_seed(0)
        else:
            torch.cuda.manual_seed(0)
        noise = torch.randn(1000, nz, 1, 1, device=device)
        with torch.no_grad():
            for i in range(10):                
                self.imgs.append(gen(noise[i*100:(i+1)*100]).detach().cpu().numpy())
        self.imgs = np.concatenate(self.imgs,axis=0).transpose(0,2,3,1)
        self.imgs = 255*(0.5*self.imgs + 0.5)
        self.imgs = self.imgs.astype("uint8")
        self.transform = transform
        self.len = len(self.imgs)
    def __getitem__(self, index):
        image = Image.fromarray(self.imgs[index])
        if self.transform is not None:
            image = self.transform(image)
        return image
    def __len__(self):
        return self.len

nz = 100
nc = 3
batch_size = 128
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5,.5,.5), (.5,.5,.5 )),])
dataset = P1("hw2_data/face/train",transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

def init_weights(m):
    if  isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, mean=0., std=0.02)
        nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0., std=0.02)        


# nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
# class Generator(nn.Module):
#     def __init__(self,nz,nc): # nz,1,1
#         super(Generator, self).__init__()
#         self.block0 = nn.Sequential(nn.ConvTranspose2d(nz,512,4,1,0,bias=False), 
#                                     nn.BatchNorm2d(512),
#                                     nn.ReLU(inplace=True)) # 512,4,4
#         self.block1 = nn.Sequential(nn.ConvTranspose2d(512,256,4,2,1,bias=False), 
#                                     nn.BatchNorm2d(256),
#                                     nn.ReLU(inplace=True)) # 256,8,8
#         self.block2 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1,bias=False), 
#                                     nn.BatchNorm2d(128),
#                                     nn.ReLU(inplace=True)) # 128,16,16
#         self.block3 = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1,bias=False), 
#                                     nn.BatchNorm2d(64),
#                                     nn.ReLU(inplace=True)) # 64,32,32
#         self.block4 = nn.Sequential(nn.ConvTranspose2d(64,nc,4,2,1,bias=False), 
#                                     nn.Tanh()) # nc*64*64
#     def forward(self,x):
#         x = self.block0(x)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         return x
# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

# nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0
# class Discriminator(nn.Module):
#     def __init__(self,nc): # nc,64,64
#         super(Discriminator, self).__init__()
#         self.block0 = nn.Sequential(nn.Conv2d(nc,64,4,2,1,bias=False),
#                                     nn.LeakyReLU(negative_slope=0.2,inplace=True)) # 64,32,32
#         self.block1 = nn.Sequential(nn.Conv2d(64,128,4,2,1,bias=False),
#                                     nn.BatchNorm2d(128),
#                                     nn.LeakyReLU(negative_slope=0.2,inplace=True)) # 128*16,16
#         self.block2 = nn.Sequential(nn.Conv2d(128,256,4,2,1,bias=False),
#                                     nn.BatchNorm2d(256),
#                                     nn.LeakyReLU(negative_slope=0.2,inplace=True)) # 256,8,8
#         self.block3 = nn.Sequential(nn.Conv2d(256,512,4,2,1,bias=False),
#                                     nn.BatchNorm2d(512),
#                                     nn.LeakyReLU(negative_slope=0.2,inplace=True)) # 512,4,4
#         self.block4 = nn.Sequential(nn.Conv2d(512,1,4,1,0,bias=False),
#                                     nn.Sigmoid()) #1*1*1
                                    
#     def forward(self,x):
#         x = self.block0(x)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.block4(x)
#         return x.view(-1)

# gen = Generator(nz,nc)
gen = Generator(1)
gen.apply(init_weights)
gen.to(device)
#dis = Discriminator(nc)
dis = Discriminator(1)
dis.apply(init_weights)
dis.to(device)
lr = 0.0002
beta1 = 0.5
criterion = nn.BCELoss()
optimG = optim.Adam(gen.parameters(),lr=lr,betas=(beta1,0.999))
optimD = optim.Adam(dis.parameters(),lr=lr,betas=(beta1,0.999))
epochs = 300
ite = 0
hbatch_size = batch_size//2

ckpt_dir = "./ckpt2-1_pei"
os.makedirs(ckpt_dir, exist_ok=True)

for ep in tqdm.tqdm(range(epochs)):
    for realImg in dataloader:
        lossD = 0.
        optimD.zero_grad()
        realImg = realImg.to(device)
        realLabel = torch.ones(batch_size).to(device)
        lossD += criterion(dis(realImg),realLabel)
        noise = torch.randn(batch_size, nz, 1, 1).to(device)
        fakeImg = gen(noise).detach()
        fakeLabel = torch.zeros(batch_size).to(device)
        lossD += criterion(dis(fakeImg),fakeLabel)
        lossD.backward()
        optimD.step()
        optimG.zero_grad()
        noise = torch.randn(batch_size, nz, 1, 1).to(device)
        fakeImg = gen(noise)
        realLabel = torch.ones(batch_size).to(device)
        MSLoss = torch.mean(torch.abs(fakeImg[:hbatch_size]-fakeImg[hbatch_size:]).sum(1).sum(1).sum(1)\
                /torch.abs(noise[:hbatch_size]-noise[hbatch_size:]).sum(1).sum(1).sum(1))
        lossG = criterion(dis(fakeImg),realLabel) + 1/(MSLoss+1e-5)
        lossG.backward()
        optimG.step()
        ite += 1
    # print("Saving")
    # torch.save(gen.state_dict(), os.path.join(ckpt_dir, f'G_{ep}.pth'))
    # torch.save(dis.state_dict(), os.path.join(ckpt_dir, f'D_{ep}.pth'))
    if ep>100:

        os.makedirs("./output_images_3", exist_ok=True)

        evalset = P1Eval(gen,nz,transform)
        print("ite: %d lossD.item(): %f lossG.item(): %f"%(ite, lossD.item(), lossG.item()))

        for i in range(1000):
            plt.imsave("./output_images_3/{0:03d}.png".format(i),evalset.imgs[i])
        batcmd="python3 -W ignore face_recog.py --image_dir output_images_3"
        result = subprocess.check_output(batcmd, shell=True)
        print("-------------------------------")
        print(str(result))
        face_recog = float(str(result)[-10:-8])
        print("face_recog: %d"%face_recog)
        print("-------------------------------")
        batcmd="python -W ignore -m pytorch_fid hw2_data/face/val output_images_3"
        result = subprocess.check_output(batcmd, shell=True)
        print(str(result))
        fid = float(str(result)[8:-3])
        print("fid: %d"%fid)
        print("-------------------------------")

        if fid < 30 and face_recog > 85:
            print("Simple Passed.")
            #state = {'stateG': gen.state_dict(),'optimG' : optimG.state_dict(),'stateD': dis.state_dict(),'optimD' : optimD.state_dict()}
            #torch.save(state, savePath) 
            torch.save(gen.state_dict(), os.path.join(ckpt_dir, f'G_{ep}_simple.pth'))
            torch.save(dis.state_dict(), os.path.join(ckpt_dir, f'D_{ep}_simple.pth'))

        if fid < 27 and face_recog > 90:
            print("Strong Passed.")
            #state = {'stateG': gen.state_dict(),'optimG' : optimG.state_dict(),'stateD': dis.state_dict(),'optimD' : optimD.state_dict()}
            #torch.save(state, savePath) 
            torch.save(gen.state_dict(), os.path.join(ckpt_dir, f'G_{ep}_strong.pth'))
            torch.save(dis.state_dict(), os.path.join(ckpt_dir, f'D_{ep}_strong.pth'))

        print("Iteration %d finished."%(ite/100))
        print("-------------------------------")
    print("Epoch %d finished."%ep)
    print("-------------------------------")

# savePath = "%d.pth"%(ite)
# state = {'stateG': gen.state_dict(),'optimG' : optimG.state_dict(),'stateD': dis.state_dict(),'optimD' : optimD.state_dict()}
# torch.save(state, savePath)

