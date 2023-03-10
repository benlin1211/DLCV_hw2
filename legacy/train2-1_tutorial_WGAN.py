# ref https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html?fbclid=IwAR0BxfJus3-pz_tMnLBbPSMzRkQxI1WGes5t3uQUtzOgtEcuC0xOTD1Z1-Q

from __future__ import print_function
#%matplotlib inline
import argparse
import glob
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torch.utils.data
import torchvision.datasets as dset

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import torch.autograd as autograd
from torch.autograd import Variable


parser = argparse.ArgumentParser(description="hw 2-1 train",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("output_dir", help="Output data location")
parser.add_argument("--mode", help="train or test", default="train")   
parser.add_argument("--pth_name") 
parser.add_argument("--ckpt_dir", help="Checkpoint location", default="ckpt2-1_tutorial_B")

args = parser.parse_args()
mode = args.mode
pth_name = args.pth_name
output_dir = args.output_dir

# Set random seed for reproducibility
manualSeed = 1
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "./hw2_data/face/train"
ckpt_dir = "./ckpt2-1_B"
os.makedirs(ckpt_dir, exist_ok=True)

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 64

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 400

# Learning rate for optimizers
lr = 0.0002
#lr = 0.00005

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
n_critic = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
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
        # transforms.CenterCrop(64),
        # transforms.Resize(64),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = FaceDataset(fnames, transform)
    return dataset

# Create the dataloader
dataset = get_dataset(dataroot)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
if torch.cuda.is_available():
    if torch.cuda.device_count()==2:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using", device)

# # Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

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

# Create the generator
netG = Generator(ngpu).to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.02.
netG.apply(weights_init)

# Print the model
# print(netG)

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

# Create the Discriminator
netD = Discriminator(ngpu).to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netD.apply(weights_init)

# Print the model
# print(netD)


if args.mode=="test":
    os.makedirs(output_dir, exist_ok=True)
    n_generate = 1000
    netG = Generator(ngpu)
    ckpt_path = os.path.join(ckpt_dir, args.pth_name)
    # ckpt_path = os.path.join("./ckpt2-1_tutorial/G_371.pth")
    
    print(f"Load model from: {ckpt_path}")
    netG.load_state_dict(torch.load(ckpt_path))
    netG = netG.to(device)
    netG.eval()

    z = torch.randn(n_generate, nz, 1, 1).to(device)
    imgs = netG(z).data
    #imgs = (imgs + 1) / 2.0

    for i in range(n_generate):
        torchvision.utils.save_image(imgs[i], os.path.join(output_dir,f'{i+1}.jpg'), normalize=True)
    
    print("Done.")



if args.mode=="train":

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # resume_path = os.path.join("ckpt2-1_tutorial", "G_371.pth")
    # print(f"Load model from: {resume_path}")
    # netG.load_state_dict(torch.load(resume_path))
    # resume_path = os.path.join("ckpt2-1_tutorial", "D_371.pth")
    # print(f"Load model from: {resume_path}")
    # netD.load_state_dict(torch.load(resume_path))
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    print("Starting Training Loop...")

    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        # ratio = 0.1 * (1-epoch/num_epochs)
        ratio = 0.1
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
                r_logits = netD(real_cpu).view(-1)
                r_logit_mean = torch.mean(r_logits)
                errD_real = -r_logit_mean 
                errD_real.backward()

                ## Train with all-fake batch

                # Classify all fake batch with D
                f_logits = netD(fake.detach()).view(-1)
                f_logit_mean = torch.mean(f_logits)
                errD_fake = f_logit_mean
                errD_fake.backward()

                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                gradient_penalty = gp(netD, device, real_samples=real_cpu, fake_samples=fake)
                #errD = -r_logit_mean + f_logit_mean + gradient_penalty
                gradient_penalty.backward()
                errD = -r_logit_mean.detach() + f_logit_mean.detach() + gradient_penalty.detach()
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

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        if True:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
                fake = (fake+1)/2
            filename = os.path.join(ckpt_dir, f'Epoch_{epoch:03d}.jpg')
            torchvision.utils.save_image(fake, filename, nrow=8, normalize=True)            
            torch.save(netG.state_dict(), os.path.join(ckpt_dir, f'G_{epoch}.pth'))
            torch.save(netD.state_dict(), os.path.join(ckpt_dir, f'D_{epoch}.pth'))

        iters += 1
                
