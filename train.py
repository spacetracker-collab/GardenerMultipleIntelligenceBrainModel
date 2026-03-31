
import torch
from model import GardnerNet

def fitness(out):
    return out.mean() + torch.std(out) - torch.var(out)

model = GardnerNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

agents = 6

for step in range(50):
    x = torch.randn(agents,32)
    seq = torch.randn(agents,10,32)
    img = torch.randn(agents,1,8,8)
    audio = torch.randn(agents,1,16)

    edge_index = torch.combinations(torch.arange(agents), r=2).T

    out = model(x, seq, img, audio, edge_index)
    fit = fitness(out)

    loss = -fit

    opt.zero_grad()
    loss.backward()

    # meta-like adaptation
    for p in model.parameters():
        if p.grad is not None:
            p.data -= 0.005 * p.grad

    opt.step()

    print(f"Step {step} Fitness: {fit.item():.4f}")
