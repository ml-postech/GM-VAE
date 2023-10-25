from torch import nn


def add_model_args(parser):
    pass


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.latent_dim = args.latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),  # (32)
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),  # (16)
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),  # (8)
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),  # (4)
            nn.Flatten(),
        )
        self.output_dim = 4 * 4 * 256

    def forward(self, x):
        feature = self.encoder(x)
        return feature


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
    
        self.latent_dim = args.latent_dim * 2
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 256, 4, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        fixed_shapes = z.size()[:-1] # z \in [1, B, 1, 2]
        z = z.reshape(-1, self.latent_dim, 1, 1)
        x = self.decoder(z)
        x = x.view(*fixed_shapes, 3, 64, 64)

        return x

