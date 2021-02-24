import math
import torch
import wandb
import torchvision
from torch import nn
from utils import generate_even_data
from models import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(dataloader, epochs, nz, batch_size, lr, nf, nc):
    # Models
    generator = Generator(nf=nf, nc=nc)
    discriminator = Discriminator(nf=nf, nc=nc)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    generator = generator.cuda()
    discriminator = discriminator.cuda()

    generator.train()
    discriminator.eval()

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr = lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = lr)

    loss = nn.BCELoss()

    writer_fake = SummaryWriter(f"logs/fake")
    writer_real = SummaryWriter(f"logs/fake")

    for epoch in range(epochs):
        print(epoch,"/",epochs)
        for data, _ in dataloader:
            data = data.cuda()
            real_labels = torch.ones(data.size()[0]).cuda()
            fake_labels = torch.zeros(data.size()[0]).cuda()

            generator_optimizer.zero_grad()

            # generate data
            noise = torch.rand(data.size()[0], nz, 1, 1).cuda()
            generated_data = generator(noise)

            img_grid_fake = torchvision.utils.make_grid(
                generated_data[:4], normalize=True
            )
            writer_fake.add_image("Fake", img_grid_fake)
            
            # Train the generator
            # We invert the labels here and don't train the discriminator because we want the generator
            # to make things the discriminator classifies as true

            generator_discriminator_out = discriminator(generated_data).view(-1)
            generator_loss = loss(generator_discriminator_out, real_labels)
            generator_loss.backward()
            generator_optimizer.step()

            # Train the discriminator on the true/generated data
            discriminator_optimizer.zero_grad()
            true_discriminator_out = discriminator(data).view(-1)
            true_discriminator_loss = loss(true_discriminator_out, real_labels)

            generator_discriminator_out = discriminator(generated_data.detach()).view(-1)
            generator_discriminator_loss = loss(generator_discriminator_out, fake_labels)
            discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()

            wandb.log({
                "generator_loss": generator_loss,
                "discriminator_loss": discriminator_loss
            })

    # save models

if __name__ == "__main__":
    hyperparameters_defaults = dict(
        batch_size = 64,
        lr = 1e-3,
        nz = 100,
    )

    run = wandb.init(project="gan", job_type="train", config = hyperparameters_defaults)
    config = wandb.config

    tfms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor()
    ])
    dataset = torchvision.datasets.MNIST("", download = True, transform = tfms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = config.batch_size, shuffle = True)

    train(dataloader, epochs=15, nz=config.nz, batch_size=config.batch_size, lr=config.lr, nf=64, nc=1)
