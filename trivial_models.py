from torch import nn

class Generator(nn.Module):

    def __init__(self, input_length):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length), int(input_length))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.dense_layer(x)
        x = self.activation(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_length):
        super(Discriminator, self).__init__()

        self.dense = nn.Linear(input_length, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        return x


