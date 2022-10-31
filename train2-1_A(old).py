# import module
import os
import glob
import argparse
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import logging
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


# setting for weight init function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# prepare for Dataset
class FaceDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

def get_dataset(directory):
    print(os.path.join(directory, '*.png'))
    fnames = glob.glob(os.path.join(directory, '*.png'))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)), #image_size = 64
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = FaceDataset(fnames, transform)
    return dataset

# Generator
class Generator(nn.Module):
    def __init__(self, nz, nc=3, ngf=64):
        super(Generator, self).__init__()
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
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
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
        self.apply(weights_init)

    def forward(self, input):
        output = self.main(input) 
        return output.view(-1)

class TrainerGAN():
    def __init__(self, config):
        self.config = config
        
        self.G = Generator(100)
        self.D = Discriminator(3)
        
        self.loss = nn.BCELoss()

        """
        NOTE FOR SETTING OPTIMIZER:

        GAN: use Adam optimizer
        WGAN: use RMSprop optimizer
        WGAN-GP: use Adam optimizer 
        """
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        
        self.dataloader = None
        
        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO, 
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')
        
        self.steps = 0
        self.device = self.config["device"]
        self.z_samples = Variable(torch.randn(64, self.config["z_dim"], 1, 1)).to(self.device)

    def print_model(self):
        print("Model A info:")
        print(self.G)
        print(self.D)
        
    def prepare_environment(self):
        """
        Use this funciton to prepare function
        """
        os.makedirs(self.config["ckpt_dir"], exist_ok=True)
        os.makedirs(self.config["ckpt_dir"], exist_ok=True)
        
        # create dataset by the above function
        dataset = get_dataset(self.config["data_dir"])
        print("Number of images:",len(dataset))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)
        
        # model preparation
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)
        self.G.train()
        self.D.train()
        
    def train(self):
        """
        Use this function to train generator and discriminator
        """
        self.prepare_environment()
        
        for e, epoch in enumerate(range(self.config["n_epoch"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            for i, data in enumerate(progress_bar):
                imgs = data.to(self.device)
                bs = imgs.size(0)

                # *********************
                # *    Train D        *
                # *********************
                z = Variable(torch.randn(bs, self.config["z_dim"], 1, 1)).to(self.device)
                r_imgs = Variable(imgs).to(self.device)
                f_imgs = self.G(z)
                r_label = torch.ones((bs)).to(self.device)
                f_label = torch.zeros((bs)).to(self.device)


                # Discriminator forwarding
                #print(r_imgs.shape)
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)

                # Loss for discriminatolsr
                r_loss = self.loss(r_logit, r_label)
                f_loss = self.loss(f_logit, f_label)
                loss_D = (r_loss + f_loss) / 2

                # Discriminator backwarding
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                if self.steps % self.config["n_critic"] == 0:
                    # Generate some fake images.
                    z = Variable(torch.randn(bs, self.config["z_dim"], 1, 1)).to(self.device)
                    f_imgs = self.G(z)

                    # Generator forwarding
                    f_logit = self.D(f_imgs)

                    # Loss for the generator.
                    loss_G = self.loss(f_logit, r_label)

                    # Generator backwarding
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()
                    
                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item())
                self.steps += 1

            self.G.eval()
            f_imgs_sample = (self.G(self.z_samples).data + 1) / 2.0
            filename = os.path.join(self.config["ckpt_dir"], f'Epoch_{epoch+1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=8)
            logging.info(f'Save some samples to {filename}.')

            # Show some images during training.
            # grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=8)
            # plt.figure(figsize=(10,10))
            # plt.imshow(grid_img.permute(1, 2, 0))
            # plt.show()

            self.G.train()

            if (e+1) % config["save_every"] == 0 or e == 0:
                # Save the checkpoints.
                torch.save(self.G.state_dict(), os.path.join(self.config["ckpt_dir"], f'G_{e}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.config["ckpt_dir"], f'D_{e}.pth'))

        logging.info('Finish training')

    def inference(self, G_path, n_generate=1000, n_output=30, show=False):

        self.G.load_state_dict(torch.load(G_path))
        self.G.to(self.device)
        self.G.eval()
        
        # Generate 1000 imgs
        z = Variable(torch.randn(n_generate, self.config["z_dim"], 1, 1)).to(self.device)
        imgs = (self.G(z).data + 1) / 2.0
        
        # Save 1000 imgs
        os.makedirs( self.config["output_dir"], exist_ok=True)
        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], os.path.join( self.config["output_dir"],f'{i+1}.jpg'))
        
        if show:
            row, col = n_output//10 + 1, 10
            grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
            plt.figure(figsize=(row, col))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="hw 2-1 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", help="Output data location")
    parser.add_argument("--data_dir", help="Training data location", default="./hw2_data/face/train")
    parser.add_argument("--mode", help="train or test", default="train")   
    parser.add_argument("--log_dir", help="Log location", default="log2-1")
    parser.add_argument("--ckpt_dir", help="Checkpoint location", default="ckpt2-1A")
    parser.add_argument("--save_every", help="Save model every k epochs", type=int, default=5)
    parser.add_argument("--batch_size", help="batch size", type=int, default=128)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--n_epoch", help="n_epoch", type=int, default=200)
    parser.add_argument("--n_critic", help="Update generater for every n epochs.", type=int, default=1)

    parser.add_argument("--z_dim", help="Latent space dimension", type=int, default=100)

    parser.add_argument("--scheduler_lr_decay_step", help="scheduler learning rate decay step ", type=int, default=1)
    parser.add_argument("--scheduler_lr_decay_ratio", help="scheduler learning rate decay ratio ", type=float, default=0.99)
    args = parser.parse_args()
    print(vars(args))
    
    same_seeds(2022)


    if torch.cuda.is_available():
        if torch.cuda.device_count()==2:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using", device)



    config = {
        "output_dir": args.output_dir,
        "data_dir": args.data_dir,
        "log_dir": args.log_dir,
        "ckpt_dir": args.ckpt_dir,

        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "n_epoch": args.n_epoch,
        "n_critic": args.n_critic,
        "z_dim": args.z_dim,
        "save_every": args.save_every,
        "device": device,
    }
    trainer = TrainerGAN(config)
    trainer.print_model()
    if args.mode == "train":
        trainer.train()
    if args.mode == "test":
        trainer.inference(f'{args.ckpt_dir}/G_114.pth',show = True) # you have to modify the path when running this line
        print("Done.")
