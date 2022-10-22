# import module
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
    """
    Input shape: (batch, in_dim)
    Output shape: (batch, 3, 64, 64)
    """
    def __init__(self, in_dim, feature_dim=64, apply_weights_init=True):
        super().__init__()
    
        #input: (batch, 100)
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),               #(batch, feature_dim * 16, 8, 8)     
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),               #(batch, feature_dim * 16, 16, 16)     
            self.dconv_bn_relu(feature_dim * 2, feature_dim),                   #(batch, feature_dim * 16, 32, 32)     
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()   
        )
        if apply_weights_init:
            self.apply(weights_init)

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),        #double height and width
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )
    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y

# Discriminator
class Discriminator(nn.Module):
    """
    Input shape: (batch, 3, 64, 64)
    Output shape: (batch)
    """
    def __init__(self, in_dim, feature_dim=64, apply_weights_init=True):
        super(Discriminator, self).__init__()
            
        #input: (batch, 3, 64, 64)
        """
        NOTE FOR SETTING DISCRIMINATOR:
        Remove last sigmoid layer for WGAN
        """
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1), #(batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(feature_dim, feature_dim * 2),                   #(batch, 3, 16, 16)
            self.conv_bn_lrelu(feature_dim * 2, feature_dim * 4),               #(batch, 3, 8, 8)
            self.conv_bn_lrelu(feature_dim * 4, feature_dim * 8),               #(batch, 3, 4, 4)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            # nn.Sigmoid() 
        )
        if apply_weights_init:
            self.apply(weights_init)
            
    def conv_bn_lrelu(self, in_dim, out_dim):
        """
        NOTE FOR SETTING DISCRIMINATOR:
        You can't use nn.Batchnorm for WGAN-GP
        Use nn.InstanceNorm2d instead
        """
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.InstanceNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1)
        # print("yy",y.shape)
        return y


class TrainerGAN():
    def __init__(self, config):
        self.config = config
        
        self.G = Generator(config["latent_dim"], feature_dim=64)
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
        self.z_samples = Variable(torch.randn(64, self.config["latent_dim"])).to(self.device)

    def print_model(self):
        print("Model B info:")
        print(self.G)
        print(self.D)
        
    def prepare_environment(self):
        """
        Use this funciton to prepare function
        """
        os.makedirs(self.config["ckpt_dir"], exist_ok=True)
        #os.makedirs(self.config["output_dir"], exist_ok=True)
        
        # create dataset by the above function
        dataset = get_dataset(self.config["data_dir"])
        print("Number of images:",len(dataset))
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)
        
        # model preparation
        self.G = self.G.to(self.device)
        self.D = self.D.to(self.device)
        self.G.train()
        self.D.train()

    def get_gradient_norm(self, model_layer):
        """
       Compute gradient norm for specific layer.
        """
        total_norm = 0
        parameters = [p for p in model_layer.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        return total_norm

    def gp(self, real_samples, fake_samples):
        """
        Calculates the gradient penalty loss for WGAN GP.
        """
        # Random weight term for interpolation between real and fake samples
        Tensor = torch.FloatTensor
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.D(interpolates).unsqueeze(1)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
        
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

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
                z = Variable(torch.randn(bs, self.config["latent_dim"])).to(self.device)

                r_imgs_origin = Variable(imgs).to(self.device) 
                f_imgs_origin = self.G(z) 
                
                if epoch < 40:
                    r_imgs = r_imgs_origin + 0.25*Variable(torch.randn(bs, 3, 64, 64)).to(self.device)
                    f_imgs = f_imgs_origin + 0.25*Variable(torch.randn(bs, 3, 64, 64)).to(self.device)
                elif epoch < 80:
                    r_imgs = r_imgs_origin + 0.1*Variable(torch.randn(bs, 3, 64, 64)).to(self.device)
                    f_imgs = f_imgs_origin + 0.1*Variable(torch.randn(bs, 3, 64, 64)).to(self.device)
                else:
                    r_imgs = r_imgs_origin
                    f_imgs = f_imgs_origin 

                r_label = torch.ones((bs)).to(self.device)
                f_label = torch.zeros((bs)).to(self.device)

                # Discriminator forwarding
                #print(r_imgs.shape)
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)

                """
                NOTE FOR SETTING DISCRIMINATOR LOSS:
                GAN: 
                    r_loss = self.loss(r_logit, r_label) 
                    f_loss = self.loss(f_logit, f_label)
                    loss_D = (r_loss + f_loss) / 2
                WGAN: 
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
                WGAN-GP: 
                    gradient_penalty = self.gp(r_imgs, f_imgs)
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
                """               

                # Loss for discriminatolsr
                gradient_penalty = self.gp(r_imgs, f_imgs)
                loss_critic = -torch.mean(r_logit) + torch.mean(f_logit)
                loss_D = loss_critic + 0.1*gradient_penalty    

                # Discriminator backwarding
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                # if i == 1:
                #     torchvision.utils.save_image(r_imgs_origin[0], os.path.join( self.config["ckpt_dir"],f'_real_img_origin_{e}.jpg'))
                #     torchvision.utils.save_image(f_imgs_origin[0], os.path.join( self.config["ckpt_dir"],f'_fake_img_origin_{e}.jpg'))
                #     torchvision.utils.save_image(r_imgs[0], os.path.join( self.config["ckpt_dir"],f'_real_img_{e}.jpg'))
                #     torchvision.utils.save_image(f_imgs[0], os.path.join( self.config["ckpt_dir"],f'_fake_img_{e}.jpg'))

                if self.steps % self.config["n_critic"] == 0 : #or loss_critic < self.config["loss_critic_criterion"]:
                    #print("update")
                    # Generate some fake images.
                    z = Variable(torch.randn(bs, self.config["latent_dim"])).to(self.device)
                    f_imgs_origin = self.G(z)
                    f_imgs = f_imgs_origin 

                    # Generator forwarding
                    f_logit = self.D(f_imgs) 
                    """
                    NOTE FOR SETTING LOSS FOR GENERATOR:
                    
                    GAN: loss_G = self.loss(f_logit, r_label)
                    WGAN: loss_G = -torch.mean(self.D(f_imgs))
                    WGAN-GP: loss_G = -torch.mean(self.D(f_imgs))
                    """
                    # Loss for the generator.
                    loss_G = -torch.mean(self.D(f_imgs))

                    # Generator backwarding
                    self.G.zero_grad()
                    loss_G.backward()
                    self.opt_G.step()
                # else:
                #     print(f"G is not updated. loss_D = {loss_D}")
                
                if self.steps % 2 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_critic=loss_critic.item(), loss_D=loss_D.item())
                    wandb.log({
                        "loss_G": loss_G.item(),
                        "loss_critic": loss_critic.item(),
                        "loss_D": loss_D.item()})
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
        z = Variable(torch.randn(n_generate, self.config["latent_dim"])).to(self.device)
        imgs = (self.G(z).data + 1) / 2.0 #??
        
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
    parser.add_argument("--ckpt_dir", help="Checkpoint location", default="ckpt2-1B")
    parser.add_argument("--save_every", help="Save model every n epochs", type=int, default=5)
    parser.add_argument("--batch_size", help="batch size", type=int, default=128)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-4)
    parser.add_argument("--n_epoch", help="n_epoch", type=int, default=200)
    parser.add_argument("--n_critic", help="Update generater for every k steps in a epoch.", type=int, default=1)
    #parser.add_argument("--loss_critic_criterion", help="Update generater when discriminator critic loss < c.", type=float, default=0)

    parser.add_argument("--latent_dim", help="Latent space dimension", type=int, default=100) #100
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

    

    config = {
        "output_dir": args.output_dir,
        "data_dir": args.data_dir,
        "ckpt_dir": args.ckpt_dir,

        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "n_epoch": args.n_epoch,
        "n_critic": args.n_critic,
        #"loss_critic_criterion": args.loss_critic_criterion,
        "latent_dim": args.latent_dim,
        "save_every": args.save_every,
        "device": device,
    }
    trainer = TrainerGAN(config)
    trainer.print_model()
    if args.mode == "train":
        wandb.init(entity="benlin1211", project="DLCV hw2-1")
        trainer.train()
    if args.mode == "test":
        
        #model_path = os.path.join(args.ckpt_dir,f'G_{args.n_epoch-1}.pth' )
        model_path = os.path.join(args.ckpt_dir,f'G_189.pth' )
        print(f"Loading from {model_path}")
        trainer.inference(model_path,show = True) # you have to modify the path when running this line
        print("Done.")
    
    #https://stats.stackexchange.com/questions/505696/when-is-my-wasserstein-gan-gp-overfitting