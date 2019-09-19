import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TimeDelayedStack(nn.Module):
    def __init__(self, dims):
        super(TimeDelayedStack, self).__init__()
        self.bi_freq_rnn = nn.GRU(dims, dims, batch_first=True, bidirectional=True)
        self.time_rnn = nn.GRU(dims, dims, batch_first=True)

    def forward(self, x_time):

        # Batch, Timesteps, Mels, Dims
        B, T, M, D = x_time.size()

        # Collapse the first two axes
        time_input = x_time.transpose(1, 2).contiguous().view(-1, T, D)
        freq_input = x_time.view(-1, M, D)

        # Run through the rnns
        x_1, _ = self.time_rnn(time_input)
        x_2_and_3, _ = self.bi_freq_rnn(freq_input)

        # Reshape the first two axes back to original
        x_1 = x_1.view(B, M, T, D).transpose(1, 2)
        x_2_and_3 = x_2_and_3.view(B, T, M, 2 * D)

        # And concatenate for output
        x_time = torch.cat([x_1, x_2_and_3], dim=3)
        return x_time


class FrequencyDelayedStack(nn.Module):
    def __init__(self, dims):
        super(FrequencyDelayedStack, self).__init__()
        self.rnn = nn.GRU(dims, dims, batch_first=True)

    def forward(self, x_time, x_freq):
        # sum the inputs
        x = x_time + x_freq

        # Batch, Timesteps, Mels, Dims
        B, T, M, D = x.size()
        # collapse the first two axes
        x = x.view(-1, M, D)

        # Through the RNN
        x, _ = self.rnn(x)
        return x.view(B, T, M, D)


class Layer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.freq_stack = FrequencyDelayedStack(dims)
        self.freq_out = nn.Linear(dims, dims)
        self.time_stack = TimeDelayedStack(dims)
        self.time_out = nn.Linear(3 * dims, dims)

    def forward(self, x):
        # unpack the input tuple
        x_time, x_freq = x

        # grab a residual for x_time
        x_time_res = x_time
        # run through the time delayed stack
        x_time = self.time_stack(x_time)
        # reshape output
        x_time = self.time_out(x_time)
        # connect time residual
        x_time = x_time + x_time_res

        # grab a residual for x_freq
        x_freq_res = x_freq
        # run through the freq delayed stack
        x_freq = self.freq_stack(x_time, x_freq)
        # reshape output TODO: is this even needed?
        x_freq = self.freq_out(x_freq)
        # connect the freq residual
        x_freq = x_freq + x_freq_res
        return [x_time, x_freq]


class MelNet(nn.Module):
    def __init__(self, batch_size, time_step, n_mels, dims, n_layers, n_mixtures):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.batch_size = batch_size
        self.time_step = time_step
        self.n_mels = n_mels
        # Input layers
        self.freq_input = nn.Linear(1, dims)
        self.time_input = nn.Linear(1, dims)

        # Main layers
        self.layers = nn.Sequential(
            *[Layer(dims) for _ in range(n_layers)]
        )

        # Output layer
        self.fc_out = nn.Linear(2 * dims, 3 * n_mixtures)

        # Print model size
        self.num_params()

    def forward(self, x):
        print("MelNet")
        print(x.shape)
        # Shift the inputs left for time-delay inputs
        # x_time = F.pad(x, [0, 0, -1, 1, 0, 0]).unsqueeze(-1)
        x_time = x.unsqueeze(-1)
        print(x_time.shape)
        # Shift the inputs down for freq-delay inputs
        # x_freq = F.pad(x, [0, 0, 0, 0, -1, 1]).unsqueeze(-1)
        x_freq = x.unsqueeze(-1)
        print(x_freq.shape)

        # Initial transform from 1 to dims
        x_time = self.time_input(x_time)
        x_freq = self.freq_input(x_freq)

        # Run through the layers
        x = (x_time, x_freq)
        x_time, x_freq = self.layers(x)

        # Get the mixture params
        x = torch.cat([x_time, x_freq], dim=-1)
        params = self.fc_out(x)
        params = params.view(self.batch_size, self.time_step, self.n_mels, self.n_mixtures, 3)

        mu = params[:, :, :, :, 0]
        sigma = torch.exp(params[:, :, :, :, 1])
        pi = F.softmax(params[:, :, :, :, 2], dim=3)
        # print(mu.shape)
        # print(sigma.shape)
        # print(pi.shape)
        # print(sum(pi[0,0,0,:]))

        return mu, sigma, pi

    def loss(self, params, target):
        print("loss")
        mu, sigma, pi = params
        m = torch.distributions.normal.Normal(mu, sigma)
        target = target.unsqueeze(-1).expand_as(mu)
        print(mu.shape)
        print(target.shape)
        print(m.log_prob(target).shape)
        log_prob = torch.sum(m.log_prob(target) * pi, dim=-1)
        # print(log_prob.shape)
        return -log_prob

    def sample(self, params):
        mu, sigma, pi = params
        m = torch.distributions.normal.Normal(mu, sigma)
        res = torch.sum(m.sample() * pi, dim=-1)
        return res

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3fM' % parameters)

if __name__ == '__main__':
    batchsize = 4
    timesteps = 10
    num_mels = 8
    dims = 512
    n_layers = 5
    n_mixtures = 10

    model = MelNet(batchsize, timesteps, num_mels, dims, n_layers, n_mixtures)

    x = torch.ones(batchsize, timesteps, num_mels)

    print("Input Shape:", x.shape)

    params = model(x)
    mu, sigma, pi = params
    # print("Output Shape", params.shape)
    print(mu.shape)
    print(sigma.shape)
    print(pi.shape)
    import math
    ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)
    data = x.unsqueeze(-1).expand_as(mu)
    print(data.shape)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / sigma)**2) / sigma

    print(ret.shape)


    m = torch.distributions.normal.Normal(mu, sigma)
    print("here")
    print(m.log_prob(data).shape)
    print(pi.shape)
    log_prob = torch.sum(m.log_prob(data) * pi, dim=-1)
    print(log_prob.shape)
