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


# Model Class Definition (Frequentist Implementation)
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


# Constants
EPOCHS     = 1
SAMPLES    = 10
BATCH_SIZE = 64
LR         = 1e-3
N_CLASSES  = 10
W, H       = 28, 28

# Dataset
transform = ToTensor()
dataset = MNIST(root="dataset", train=True, download=True, transform=transform)
loader = DataLoader(dataset,
    shuffle=True, batch_size=BATCH_SIZE,
    num_workers=4, pin_memory=True,
)

# Model and Optimizer
model = MLP(W * H, 512, N_CLASSES).cuda() # Frequentist
optim = Adam(model.parameters(), lr=LR)   # Adam Optimizer

# Main Loop
for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    tot_loss = 0.0
    tot_acc = 0.0

    # Batch Loop
    pbar = tqdm(loader, desc="Batch")
    for img, label in pbar:
        img, label = img.float().cuda(), label.long().cuda()

        # Reset Grads
        optim.zero_grad()
        
        # Loss Computation
        prediction = model(img.view(img.size(0), -1))
        loss = F.nll_loss(prediction, label, reduction="sum")
        acc = (torch.argmax(prediction, dim=1) == label).sum()
        
        # Weights Update
        loss.backward()
        optim.step()

        # Display
        tot_loss += loss.item() / len(loader)
        tot_acc += acc.item() / len(dataset) * 100
        
        pbar.set_postfix(loss=tot_loss, acc=f"{tot_acc:.2f}%")

# Convertion to Bayesian
bmodel = to_bayesian(model, delta=0.05)

tot_loss = 0.0
tot_nll = 0.0
tot_log_prior = 0.0
tot_log_variational_posterior = 0.0
tot_acc = 0.0

# Batch Loop Bayesian Eval
pbar = tqdm(loader, desc="Bayesian Eval")
for img, label in pbar:
    img, label = img.float().cuda(), label.long().cuda()

    # Setup Outputs
    prediction = torch.zeros(SAMPLES, img.size(0), 10).cuda()
    log_prior = torch.zeros(SAMPLES).cuda()
    log_variational_posterior = torch.zeros(SAMPLES).cuda()

    # Sample Loop (VI)
    for s in range(SAMPLES):
        prediction[s] = bmodel(img.view(img.size(0), -1))
        log_prior[s] = bmodel.log_prior()
        log_variational_posterior[s] = bmodel.log_variational_posterior()

    # Loss Computation
    log_prior = log_prior.mean()
    log_variational_posterior = log_variational_posterior.mean()
    nll = F.nll_loss(prediction.mean(0), label, reduction="sum")
    loss = (log_variational_posterior - log_prior) / len(loader) + nll
    acc = (torch.argmax(prediction.mean(0), dim=1) == label).sum()

    # Display
    nb = len(loader)
    tot_loss += loss.item() / nb
    tot_nll += nll.item() / nb
    tot_log_prior += log_prior.item() / nb
    tot_log_variational_posterior += log_variational_posterior.item() / nb
    tot_acc += acc.item() / len(dataset) * 100

    nb = len(loader)
    pbar.set_postfix(
        loss=tot_loss,
        nll=tot_nll,
        log_prior=tot_log_prior,
        log_variational_posterior=tot_log_variational_posterior,
        acc=f"{tot_acc:.2f}%"
    )

# Main Loop
for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    tot_loss = 0.0
    tot_nll = 0.0
    tot_log_prior = 0.0
    tot_log_variational_posterior = 0.0
    tot_acc = 0.0

    # Batch Loop Bayesian Train
    pbar = tqdm(loader, desc="Bayesian")
    for img, label in pbar:
        img, label = img.float().cuda(), label.long().cuda()

        # Setup Outputs
        prediction = torch.zeros(SAMPLES, img.size(0), 10).cuda()
        log_prior = torch.zeros(SAMPLES).cuda()
        log_variational_posterior = torch.zeros(SAMPLES).cuda()

        # Sample Loop (VI)
        for s in range(SAMPLES):
            prediction[s] = bmodel(img.view(img.size(0), -1))
            log_prior[s] = bmodel.log_prior()
            log_variational_posterior[s] = bmodel.log_variational_posterior()

        # Loss Computation
        log_prior = log_prior.mean()
        log_variational_posterior = log_variational_posterior.mean()
        nll = F.nll_loss(prediction.mean(0), label, reduction="sum")
        loss = (log_variational_posterior - log_prior) / len(loader) + nll
        acc = (torch.argmax(prediction.mean(0), dim=1) == label).sum()

        # Weights Update
        loss.backward()
        optim.step()

        # Display
        nb = len(loader)
        tot_loss += loss.item() / nb
        tot_nll += nll.item() / nb
        tot_log_prior += log_prior.item() / nb
        tot_log_variational_posterior += log_variational_posterior.item() / nb
        tot_acc += acc.item() / len(dataset) * 100

        nb = len(loader)
        pbar.set_postfix(
            loss=tot_loss,
            nll=tot_nll,
            log_prior=tot_log_prior,
            log_variational_posterior=tot_log_variational_posterior,
            acc=f"{tot_acc:.2f}%"
        )
