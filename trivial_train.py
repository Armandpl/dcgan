import math
import torch
import wandb
from torch import nn
from utils import generate_even_data
from models import Generator, Discriminator

def train(max_int = 128, batch_size = 16, training_steps = 200):
    input_length = int(math.log(max_int, 2))

    # Models
    generator = Generator(input_length)
    discriminator = Discriminator(input_length)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr = 0.001)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 0.001)

    loss = nn.BCELoss()

    for i in range(training_steps):
        generator_optimizer.zero_grad()

        # generate data
        noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
        generated_data = generator(noise)
        gd = torch.where(generated_data>0.5, 1, 0)
        res = int("".join(str(int(x)) for x in list(gd[0])), 2)
        print(res)
        
        # generate examples of even real data
        true_labels, true_data = generate_even_data(max_int, batch_size=batch_size)
        true_labels = torch.Tensor(true_labels).float()
        true_data = torch.Tensor(true_data).float()

        # Train the generator
        # We invert the labels here and don't train the discriminator because we generator
        # to make things the discriminator classifies as true

        generator_discriminator_out = discriminator(generated_data)
        generator_loss = loss(generator_discriminator_out, true_labels.unsqueeze(1))
        generator_loss.backward()
        generator_optimizer.step()

        # Train the discriminator on the true/generated data
        discriminator_optimizer.zero_grad()
        true_discriminator_out = discriminator(true_data)
        true_discriminator_loss = loss(true_discriminator_out, true_labels.unsqueeze(1))

        generator_discriminator_out = discriminator(generated_data.detach())
        generator_discriminator_loss = loss(generator_discriminator_out, torch.zeros(batch_size, 1))
        discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        wandb.log({
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss
        })

if __name__ == "__main__":
    wandb.init(project="gan", job_type="train")
    train()
