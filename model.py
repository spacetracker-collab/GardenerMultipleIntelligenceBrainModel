
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class InterpersonalGNN(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x

class GardnerNet(nn.Module):
    def __init__(self, dim=32, hidden=64):
        super().__init__()

        self.logical = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU())
        self.linguistic = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=2), num_layers=1)

        self.spatial = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.musical = nn.Conv1d(1, 4, kernel_size=3, padding=1)

        self.kinesthetic = nn.Linear(dim, hidden)
        self.interpersonal = InterpersonalGNN(dim, hidden)
        self.intrapersonal = nn.GRU(dim, hidden, batch_first=True)
        self.naturalistic = nn.Linear(dim, hidden)

        self.fusion = nn.Linear(hidden*6 + 4*4 + 4*4, 1)

    def forward(self, x, seq, img, audio, edge_index):
        logical = self.logical(x)

        ling = self.linguistic(seq).mean(dim=1)

        spatial = self.spatial(img).view(x.size(0), -1)

        musical = self.musical(audio).view(x.size(0), -1)

        kin = torch.relu(self.kinesthetic(x))

        inter = self.interpersonal(x, edge_index)

        _, h = self.intrapersonal(seq)
        intra = h[-1]

        nat = torch.relu(self.naturalistic(x))

        combined = torch.cat([logical, ling, spatial, musical, kin, inter, intra, nat], dim=-1)
        return self.fusion(combined)
