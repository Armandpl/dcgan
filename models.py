from torch import nn

class Generator(nn.Module):
    
    # nc: number channels
    # nz: random input size
    # nf: number generator features maps
    def __init__(self, nc=1, nz=100, nf=64):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d( in_channels = nz, out_channels = nf * 8, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
            # state size. (nf*8) x 4 x 4
            nn.ConvTranspose2d(nf * 8, nf * 4, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            # state size. (nf*4) x 8 x 8
            nn.ConvTranspose2d( nf * 4, nf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            # state size. (nf*2) x 16 x 16
            nn.ConvTranspose2d( nf * 2, nf, 4, 2, 1, bias = False),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            # state size, (nf) x 32 x 32
            nn.ConvTranspose2d( nf, nc, 4, 2, 1, bias = False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    def __init__(self, nc=1, nf=64):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf) x 32 x 32,
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf*2) x 16 x 16
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf*4) x 8 x 8
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nf*8) x 4 x 4
            nn.Conv2d(nf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )            

    def forward(self, x):
        return self.layers(x)


