from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from bayeformers import to_bayesian
import bayeformers.nn as bnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, n_classes: int) -> None:
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes), nn.Softmax(),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.mlp(input)


# Get bayesian model
model = to_bayesian(MLP(28 * 28, 512, 10)).cuda()

epochs = 50
samples = 10
batch_size = 32
lr = 1e-3

dataset = MNIST(
    root="dataset", train=True, download=True, transform=ToTensor()
)
loader = DataLoader(
    dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
)

optim = Adam(model.parameters(), lr=lr)

for epoch in tqdm(range(epochs), desc="Epoch"):
    pbar = tqdm(loader, desc="Batch")
    for img, label in pbar:
        img, label = img.float().cuda(), label.long().cuda()

        prediction = torch.zeros(samples, img.size(0), 10).cuda()
        log_prior = torch.zeros(samples).cuda()
        log_variational_posterior = torch.zeros(samples).cuda()

        for s in tqdm(range(samples), desc="Sample"):
            prediction[s] = model(img.view(img.size(0), -1))
            log_prior[s] = model.log_prior()
            log_variational_posterior[s] = model.log_variational_posterior()

        log_prior = log_prior.mean()
        log_variational_posterior = log_variational_posterior.mean()
        nll = F.nll_loss(prediction.mean(0), label, size_average=True)

        loss = log_variational_posterior - log_prior / len(loader) + nll

        loss.backward()
        pbar.set_postfix(loss=loss.item() / len(loader))