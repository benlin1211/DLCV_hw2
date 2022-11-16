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
import wandb

# seed setting
def same_seeds(seed):
    # Python built-in random module
    # random.seed(seed)
    # # Numpy
    # np.random.seed(seed)
    # # Torch
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)


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
        transforms.CenterCrop(64),
        transforms.Resize(64),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
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
        return self.main(input)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="hw 2-1 train",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("output_dir", help="Output data location")
    parser.add_argument("--data_dir", help="Training data location", default="./hw2_data/face/train")
    parser.add_argument("--mode", help="train or test", default="train")   
    parser.add_argument("--pth_name")  
    parser.add_argument("--ckpt_dir", help="Checkpoint location", default="ckpt2-1A")
    parser.add_argument("--save_every", help="Save model every k epochs", type=int, default=1)
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--learning_rate_D", help="learning rate", type=float, default=2e-4)
    parser.add_argument("--learning_rate_G", help="learning rate", type=float, default=2e-4)
    parser.add_argument("--n_epoch", help="n_epoch", type=int, default=300)
    parser.add_argument("--n_critic", help="Update discriminator for every n epochs.", type=int, default=1)

    parser.add_argument("--z_dim", help="Latent space dimension", type=int, default=100)

    args = parser.parse_args()
    print(vars(args))
    
    same_seeds(1)


    # if torch.cuda.is_available():
    #     if torch.cuda.device_count()==2:
    #         device = torch.device("cuda:1")
    #     else:
    #         device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    # print("Using", device)
    device = torch.device("cuda")
    print("Using", device)

    # Root directory for dataset
    same_seeds(123)
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
    ngf = 64
    # Size of feature maps in discriminator
    # ndf = 64
    ndf = 64
    image_size = 64
    nc = 3


    if args.mode=="test":
        os.makedirs(output_dir, exist_ok=True)
        n_generate = 1000
        
        netG = Generator(nz, ngf, nc) 
        ckpt_path = os.path.join(ckpt_dir, args.pth_name)
        print(f"Load model from: {ckpt_path}")
        netG.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
        netG = netG.to(device)
        netG.eval()

        z = torch.randn(n_generate, nz, 1, 1).to(device)
        imgs = netG(z).data 

        imgs = (imgs + 1) / 2.0

        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], os.path.join(output_dir,'{0:03d}.png'.format(i)), normalize=True)
       
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

        # resume_path = os.path.join("ckpt2-1_pei", "G_0.pth")
        # print(f"Load model from: {resume_path}")
        # netG.load_state_dict(torch.load(resume_path))
        # resume_path = os.path.join("ckpt2-1_pei", "D_0.pth")
        # print(f"Load model from: {resume_path}")
        # netD.load_state_dict(torch.load(resume_path))        

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        # real_label = 1.
        # fake_label = 0.
        real_label = 0.9
        fake_label = 0.1

        # Setup Adam optimizers for both G and D
        optimizerD = optim.AdamW(netD.parameters(), lr=lr_D, betas=(beta1, 0.999))
        optimizerG = optim.AdamW(netG.parameters(), lr=lr_G, betas=(beta1, 0.999))


        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print("Starting Training Loop...")
        
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            #ratio = 0.01 * (1-epoch/num_epochs)
            
            # if epoch < 50:
            #     ratio = 0.2
            # elif epoch < 100:
            #     ratio = 0.15
            # elif epoch < 150:
            #     ratio = 0.1
            # elif epoch < 200:
            #     ratio = 0.05
            # else:
            #     ratio = 0
            #ratio = 1
            ratio = 0.01
            #print("ratio",ratio)
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data.to(device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output = netD(real_cpu + ratio*torch.normal(0, 1, (b_size, 3, image_size, image_size)).to(device)).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                # Generate fake image batch with G
                fake = netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach() + ratio*torch.normal(0, 1, (b_size, 3, image_size, image_size)).to(device)).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                if i % n_critic == 0:

                    netG.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output = netD(fake).view(-1)
                    # Calculate G's loss based on this output
                    errG = criterion(output, label)
                    # Calculate gradients for G
                    errG.backward()
                    D_G_z2 = output.mean().item()
                    # Update G
                    optimizerG.step()


                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, num_epochs, i, len(dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (i % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = netG(fixed_noise).detach().cpu()
                        fake = (fake+1)/2
                        
                    filename = os.path.join(ckpt_dir, f'Epoch_{epoch:03d}.png')
                    torchvision.utils.save_image(fake, filename, nrow=8,  normalize=True)


                iters += 1
                    

            if (epoch+1) % save_every == 0 or epoch == 0:
                # Save the checkpoints.
                torch.save(netG.state_dict(), os.path.join(ckpt_dir, f'G_{epoch}.pth'))
                torch.save(netD.state_dict(), os.path.join(ckpt_dir, f'D_{epoch}.pth'))

        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(ckpt_dir, 'loss curve'))
            