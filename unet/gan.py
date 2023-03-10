
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Size of feature maps in discriminator
        ndf = 64
        nc=1
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
            nn.AdaptiveAvgPool2d(1)
            # state size. (ndf*8) x 4 x 4
            
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),

            # nn.Sigmoid()
        )
        self.data_ebd = nn.Linear(6, 256)
        self.condition_ebd = nn.Linear(7, 256)
        self.out = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, data, condition):
        data = self.data_ebd(data)
        condition = self.condition_ebd(condition)

        # img = self.pool(self.conv(img)).view(data.shape[0], -1)
        img = self.main(img)
        concat_x = torch.cat((img.flatten(1), data, condition), 1)      
        return self.out(concat_x)

# Generator Code

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Number of channels in the training images. For color images this is 3
        nc = 1

        # Size of z latent vector (i.e. size of generator input)
        nz = 256

        # Size of feature maps in generator
        ngf = 256
        self.input_layer = nn.Linear(7, 256)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
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
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d( ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d( ngf // 2, ngf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 4),
            nn.ReLU(True),
            # state size. (ngf/2) x 128 x 128
            nn.ConvTranspose2d( ngf // 4, ngf // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 8),
            nn.ReLU(True),
        )
        self.img_out = nn.Sequential(
            nn.ConvTranspose2d( ngf // 8, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )
        self.data_out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(ngf // 8, 32),  ####128  ???  512???   print
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x).view(x.shape[0], 256, 1, 1)
        x = self.main(x)
        img = self.img_out(x)
        data = self.data_out(x)
        return img, data



class ConcatLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_ebd = nn.Linear(7, 128)
        # self.conv = nn.Conv2d(in_dim, 128, kernel_size=1, bias=False)
        # self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),        
            # nn.Linear(512, 6),
            # nn.Sigmoid()
        )

    def forward(self, data):
        data = self.data_ebd(data)
        # img = self.pool(self.conv(img)).view(data.shape[0], -1)
        concat_x = torch.cat((z, data), 1)
        return self.fc(concat_x)





