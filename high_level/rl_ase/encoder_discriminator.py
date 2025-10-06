import torch
import torch.nn as nn
import torch.nn.functional as F

class RunningNormalizer(nn.Module):
    def __init__(self, size, epsilon=1e-5):
        super(RunningNormalizer, self).__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('var', torch.ones(size))
        self.register_buffer('count', torch.tensor(epsilon))

    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.copy_(tot_count)

    def normalize(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)


class EncoderDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_units, encoder_output_dim):
        super(EncoderDiscriminator, self).__init__()

        self.normalizer = RunningNormalizer(input_dim)

        self.layers = nn.ModuleList()
        self.num_actions = input_dim // 4

        for hidden_unit in hidden_units:
            self.layers.append(nn.Linear(input_dim, hidden_unit))
            input_dim = hidden_unit
        
        self.encoder_mean = nn.Linear(input_dim, encoder_output_dim)
        self.discriminator_output = nn.Linear(input_dim, 1)

    def forward(self, x, discriminator):
        x = x.float()

        self.normalizer.update(x)

        x = self.normalizer.normalize(x)

        for layer in self.layers:
            x = F.relu(layer(x))
        
        if discriminator:
            x = torch.sigmoid(self.discriminator_output(x))
        
        else:
            x = self.encoder_mean(x)
            x = F.normalize(x, p=2, dim=1)
        
        return x
