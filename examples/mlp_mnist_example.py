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
            nn.Linear(hidden,      hidden), nn.ReLU(),
            nn.Linear(hidden,   n_classes), nn.LogSoftmax(dim=1),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.mlp(input)


EPOCHS     = 50
SAMPLES    = 10
BATCH_SIZE = 32
LR         = 1e-3
N_CLASSES  = 10
W, H       = 28, 28

transform = ToTensor()
dataset = MNIST(root="dataset", train=True, download=True, transform=transform)
loader = DataLoader(dataset,
    shuffle=True, batch_size=BATCH_SIZE,
    num_workers=4, pin_memory=True,
)

model = MLP(W * H, 512, N_CLASSES)
model = to_bayesian(model).cuda()
optim = Adam(model.parameters(), lr=LR)

for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    pbar = tqdm(loader, desc="Batch")
    for img, label in pbar:
        img, label = img.float().cuda(), label.long().cuda()

        prediction = torch.zeros(SAMPLES, img.size(0), 10).cuda()
        log_prior = torch.zeros(SAMPLES).cuda()
        log_variational_posterior = torch.zeros(SAMPLES).cuda()

        for s in range(SAMPLES):
            prediction[s] = model(img.view(img.size(0), -1))
            log_prior[s] = model.log_prior()
            log_variational_posterior[s] = model.log_variational_posterior()

        log_prior = log_prior.mean()
        log_variational_posterior = log_variational_posterior.mean()
        nll = F.nll_loss(prediction.mean(0), label, reduction="mean")

        loss = log_variational_posterior - log_prior / len(loader) + nll

        loss.backward()
        pbar.set_postfix(loss=loss.item() / len(loader))