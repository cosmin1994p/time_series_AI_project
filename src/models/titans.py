import torch.nn as nn
from titans_pytorch import NeuralMemory

class TitansForecastModel(nn.Module):
    def __init__(self, input_len=168, output_len=24, dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_len, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.memory = NeuralMemory(dim=dim, chunk_size=32)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim, output_len)
        )

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.memory(x.unsqueeze(1))
        x = x.squeeze(1)
        return self.decoder(x)
