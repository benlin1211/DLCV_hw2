import argparse
import glob
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch.autograd as autograd
from torch.autograd import Variable
import wandb

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
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = FaceDataset(fnames, transform)
    return dataset


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Generator Code

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
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
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def gp(netD, device, real_samples, fake_samples):
    """
    Calculates the gradient penalty loss for WGAN GP.
    """
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.FloatTensor
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = Variable(interpolates, requires_grad=True)
    d_interpolates = netD(interpolates).reshape(real_samples.shape[0], 1) 

    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="hw 2-1 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", help="Output data location")
    parser.add_argument("--data_dir", help="Training data location", default="./hw2_data/face/train")
    parser.add_argument("--mode", help="train or test", default="train")   
    parser.add_argument("--pth_name") 
    parser.add_argument("--ckpt_dir", help="Checkpoint location", default="ckpt2-1B_128")
    parser.add_argument("--save_every", help="Save model every k epochs", type=int, default=1)
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--learning_rate_D", help="learning rate", type=float, default=2e-4)
    parser.add_argument("--learning_rate_G", help="learning rate", type=float, default=2e-4)
    parser.add_argument("--n_epoch", help="n_epoch", type=int, default=120)
    parser.add_argument("--n_critic", help="Update generater for every n epochs.", type=int, default=1)

    parser.add_argument("--z_dim", help="Latent space dimension", type=int, default=100)

    args = parser.parse_args()
    print(vars(args))
    
    same_seeds(999)


    if torch.cuda.is_available():
        if torch.cuda.device_count()==2:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using", device)

    # Root directory for dataset
    same_seeds(1211)
    output_dir = args.output_dir
    ckpt_dir = args.ckpt_dir
    data_dir = args.data_dir
    batch_size =  args.batch_size
    save_every = args.save_every
    nz = args.z_dim
    num_epochs = args.n_epoch
    lr_D = args.learning_rate_D
    lr_G = args.learning_rate_G
    n_critic = args.n_critic

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Size of feature maps in generator
    #ngf = 64
    ngf = 128
    # Size of feature maps in discriminator
    #ndf = 64
    ndf = 128
    image_size = 64
    nc = 3

    if args.mode=="test":
        os.makedirs(output_dir, exist_ok=True)
        n_generate = 1000
        netG = Generator(nz, ngf, nc)
        ckpt_path = os.path.join(ckpt_dir, args.pth_name)
        print(f"Load model from: {ckpt_path}")
        netG.load_state_dict(torch.load(ckpt_path))
        netG = netG.to(device)
        netG.eval()

        z = torch.randn(n_generate, nz, 1, 1).to(device)
        imgs = netG(z).data
        imgs = (imgs + 1) / 2.0

        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], os.path.join(output_dir,f'{i+1}.jpg'))
        
        print("Done.")
        
    
    if args.mode=="train":

        os.makedirs(ckpt_dir, exist_ok=True)
        dataset = get_dataset(data_dir)
        print("Number of images:",len(dataset))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        # Create the Generator
        netG = Generator(nz, ngf, nc).to(device)
        netG.apply(weights_init)
        print(netG)

        # Create the Discriminator
        netD = Discriminator(nc, ndf).to(device)
        netD.apply(weights_init)
        print(netD)

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.AdamW(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
        optimizerG = optim.AdamW(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))


        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        wandb.init(entity="benlin1211", project="DLCV hw2-1")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            # ratio = 0.1 * (1-epoch/num_epochs)
            ratio = 0.02
            # print("ratio",ratio)

            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch

                real_cpu = data.to(device)
                b_size = real_cpu.size(0)
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)   

                if i % n_critic == 0:
                    netD.zero_grad()
                    # Format batch
                    # Forward pass real batch through D
                    r_logits = netD(real_cpu + ratio*torch.randn(b_size, 3, image_size, image_size).to(device)).view(-1)
                    r_logit_mean = torch.mean(r_logits)

                    ## Train with all-fake batch

                    # Classify all fake batch with D
                    f_logits = netD(fake.detach() + ratio*torch.randn(b_size, 3, image_size, image_size).to(device)).view(-1)
                    f_logit_mean = torch.mean(f_logits)

                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    gradient_penalty = gp(netD, device, real_samples=real_cpu, fake_samples=fake)
                    errD = -r_logit_mean + f_logit_mean + gradient_penalty
                    errD.backward()

                    # Update D
                    optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                # Since we just updated D, perform another forward pass of all-fake batch through D
                f_logits = netD(fake).view(-1)

                # Calculate G's loss based on this output
                errG = -torch.mean(f_logits)
                # Calculate gradients for G
                errG.backward()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f| Loss_G: %.4f| r_logits: %.4f| f_logits:%.4f| gp:%.4f' 
                        % (epoch, num_epochs, i, len(dataloader),
                            errD.item(), errG.item(), r_logit_mean.item(), f_logit_mean.item(), gradient_penalty))
                    wandb.log({
                        "loss_G": errG.item(),
                        "loss_D": errD.item(),
                        "r_logit": r_logit_mean.item(), 
                        "f_logit": f_logit_mean.item(),
                        "GP":gradient_penalty
                    })
                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                        fake = (fake+1)/2
                    filename = os.path.join(ckpt_dir, f'Epoch_{epoch:03d}.jpg')
                    torchvision.utils.save_image(fake, filename, nrow=8)
                    def wandb_record_img(image_array, caption):
                        image = wandb.Image(image_array, caption=caption)
                        wandb.log({caption: image})
                    
                    wandb_record_img(fake, "f_imgs")

                iters += 1
                    

            if (epoch+1) % save_every == 0 or epoch == 0:
                # Save the checkpoints.
                torch.save(netG.state_dict(), os.path.join(ckpt_dir, f'G_{epoch}.pth'))
                torch.save(netD.state_dict(), os.path.join(ckpt_dir, f'D_{epoch}.pth'))