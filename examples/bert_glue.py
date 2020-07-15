from argparse import ArgumentParser
from bayeformers import to_bayesian
from collections import namedtuple

from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import GlueDataset
from transformers import GlueDataTrainingArguments

from transformers.data.data_collator import default_data_collator as collate

from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from tqdm import tqdm
from typing import Dict

import os
import bayeformers.nn as bnn
import torch
import torch.nn as nn
import torch.nn.functional as F


parser = ArgumentParser()
parser.add_argument("--exp",           type=str,   default="exp",       help="experience name for logs")
parser.add_argument("--epochs",        type=int,   default=3,           help="epochs")
parser.add_argument("--batch_size",    type=int,   default=8,           help="batch size")
parser.add_argument("--learning_rate", type=float, default=2e-5,        help="learning rate")
parser.add_argument("--samples",       type=int,   default=10,          help="samples")
parser.add_argument("--device",        type=str,   default="cuda:0",    help="device (cpu, cuda:0, ..., cuda:n)")
parser.add_argument("--delta",         type=float, default=0.05,        help="delta rho MPOED initialization")
parser.add_argument("--freeze",                    action="store_true", help="freeze bert base mu weights")

args = parser.parse_args()


LOGS           = "logs"
EXP            = args.exp
MODEL_NAME     = "distilbert-base-uncased"
TASK_NAME      = "MRPC"
N_LABELS       = 2
MAX_SEQ_LENGTH = 128
DATA_DIR       = os.path.join("./dataset/glue/data", TASK_NAME)
LOADER_OPTIONS = { "num_workers": 4, "pin_memory": True }
EPOCHS         = args.epochs
BATCH_SIZE     = args.batch_size
LR             = args.learning_rate
WEIGHT_DECAY   = 0
ADAM_EPSILON   = 1e-8
N_WARMUP_STEPS = 0
MAX_GRAD_NORM  = 1
SAMPLES        = args.samples
DEVICE         = args.device
FREEZE         = args.freeze
DELTA          = args.delta


class Report:
    def __init__(self) -> None:
        self.total = 0.0
        self.acc = 0.0
        self.nll = 0.0
        self.log_prior = 0.0
        self.log_variational_posterior = 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.acc = 0.0
        self.nll = 0.0
        self.log_prior = 0.0
        self.log_variational_posterior = 0.0


def dic2cuda(dic: Dict) -> Dict:
    for key, value in dic.items():
        if isinstance(value, torch.Tensor):
            dic[key] = value.to(DEVICE)
    return dic


os.makedirs(LOGS, exist_ok=True)
writer = SummaryWriter(os.path.join(LOGS, EXP), filename_suffix="bayeformers_bert_glue")

config    = AutoConfig.from_pretrained(MODEL_NAME, num_labels=N_LABELS, finetuning_task=TASK_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

o_model                  = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
b_model                  = to_bayesian(o_model, delta=DELTA, freeze=FREEZE)
b_model.model.classifier = bnn.Linear.from_frequentist(o_model.classifier)
b_model                  = b_model.to(DEVICE)

glue = GlueDataTrainingArguments(TASK_NAME, data_dir=DATA_DIR, max_seq_length=MAX_SEQ_LENGTH)

train_dataset = GlueDataset(glue, tokenizer=tokenizer)
test_dataset  = GlueDataset(glue, tokenizer=tokenizer, mode="dev")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate, **LOADER_OPTIONS)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, **LOADER_OPTIONS)

params_decay    = [param for name, param in b_model.named_parameters() if name     in ["bias", "LayerNorm.weight"]]
params_no_decay = [param for name, param in b_model.named_parameters() if name not in ["bias", "LayerNorm.weight"]]
parameters      = [
    { "params": params_decay,    "weight_decay": WEIGHT_DECAY },
    { "params": params_no_decay, "weight_decay": 0.0 },
]

criterion = nn.CrossEntropyLoss(reduction="sum").to(DEVICE)
optim     = AdamW(parameters, lr=LR, eps=ADAM_EPSILON)
scheduler = get_linear_schedule_with_warmup(optim, N_WARMUP_STEPS, EPOCHS)

train_report, test_report = Report(), Report(), Report()
for epoch in tqdm(range(EPOCHS), desc="Epoch"):

    # ============================ TRAIN ======================================
    b_model.train()
    train_report.reset()
    
    pbar = tqdm(train_loader, desc="Train")
    for inputs in pbar:
        inputs = dic2cuda(inputs)
        labels = inputs["labels"]
        optim.zero_grad()

        B = inputs["input_ids"].size(0)
        logits = torch.zeros(SAMPLES, B, N_LABELS).to(DEVICE)
        log_prior = torch.zeros(SAMPLES, B).to(DEVICE)
        log_variational_posterior = torch.zeros(SAMPLES, B).to(DEVICE)

        for sample in range(SAMPLES):
            logits[sample] = b_model(**inputs)[1]
            log_prior[sample] = b_model.log_prior()
            log_variational_posterior[sample] = b_model.log_variational_posterior()

        nll = criterion(logits.mean(0), labels)
        log_prior = log_prior.mean()
        log_variational_posterior = log_variational_posterior.mean()
        loss = (log_variational_posterior - log_prior) / len(train_loader) + nll
        acc = (torch.argmax(logits.mean(0), dim=1) == labels).sum()

        loss.backward()
        nn.utils.clip_grad_norm(b_model.parameters(), MAX_GRAD_NORM)
        optim.step()
        scheduler.step()

        train_report.total += loss.item() / len(train_loader)
        train_report.nll += nll.item() / len(train_loader)
        train_report.log_prior += log_prior.item() / len(train_loader)
        train_report.log_variational_posterior += log_variational_posterior.item() / len(train_loader)
        train_report.acc += acc.item() / len(train_dataset) * 100

        pbar.set_postfix(
            total=train_report.total,
            nll=train_report.nll,
            log_prior=train_report.log_prior,
            log_variational_posterior=train_report.log_variational_posterior,
            acc=train_report.acc,
        )

    writer.add_scalar("train_total",                     train_report.total,                     epoch)
    writer.add_scalar("train_nll",                       train_report.nll,                       epoch)
    writer.add_scalar("train_log_prior",                 train_report.log_prior,                 epoch)
    writer.add_scalar("train_log_variational_posterior", train_report.log_variational_posterior, epoch)
    writer.add_scalar("train_acc",                       train_report.acc,                       epoch)

    # ============================ TEST =======================================
    b_model.eval()
    test_report.reset()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Test")
        for inputs in pbar:
            inputs = dic2cuda(inputs)
            labels = inputs["labels"]

            B = inputs["input_ids"].size(0)
            logits = torch.zeros(SAMPLES, B, N_LABELS).to(DEVICE)
            log_prior = torch.zeros(SAMPLES, B).to(DEVICE)
            log_variational_posterior = torch.zeros(SAMPLES, B).to(DEVICE)

            for sample in range(SAMPLES):
                logits[sample] = b_model(**inputs)[1]
                log_prior[sample] = b_model.log_prior()
                log_variational_posterior[sample] = b_model.log_variational_posterior()

            nll = criterion(logits.mean(0), labels)
            log_prior = log_prior.mean()
            log_variational_posterior = log_variational_posterior.mean()
            loss = (log_variational_posterior - log_prior) / len(train_loader) + nll
            acc = (torch.argmax(logits.mean(0), dim=1) == labels).sum()

            test_report.total += loss.item() / len(train_loader)
            test_report.nll += nll.item() / len(train_loader)
            test_report.log_prior += log_prior.item() / len(train_loader)
            test_report.log_variational_posterior += log_variational_posterior.item() / len(train_loader)
            test_report.acc += acc.item() / len(train_dataset) * 100

            pbar.set_postfix(
                total=test_report.total,
                nll=test_report.nll,
                log_prior=test_report.log_prior,
                log_variational_posterior=test_report.log_variational_posterior,
                acc=test_report.acc,
            )

    writer.add_scalar("test_total",                     test_report.total,                     epoch)
    writer.add_scalar("test_nll",                       test_report.nll,                       epoch)
    writer.add_scalar("test_log_prior",                 test_report.log_prior,                 epoch)
    writer.add_scalar("test_log_variational_posterior", test_report.log_variational_posterior, epoch)
    writer.add_scalar("test_acc",                       test_report.acc,                       epoch)

# ============================ EVALUTATION ====================================
b_model.eval()
test_report.reset()

with torch.no_grad():
    pbar = tqdm(test_loader, desc="Eval")
    for inputs in pbar:
        inputs = dic2cuda(inputs)
        labels = inputs["labels"]

        B = inputs["input_ids"].size(0)
        logits = torch.zeros(SAMPLES, B, N_LABELS).to(DEVICE)
        log_prior = torch.zeros(SAMPLES, B).to(DEVICE)
        log_variational_posterior = torch.zeros(SAMPLES, B).to(DEVICE)

        for sample in range(SAMPLES):
            logits[sample] = b_model(**inputs)[1]
            log_prior[sample] = b_model.log_prior()
            log_variational_posterior[sample] = b_model.log_variational_posterior()

        nll = criterion(logits.mean(0), labels)
        log_prior = log_prior.mean()
        log_variational_posterior = log_variational_posterior.mean()
        loss = (log_variational_posterior - log_prior) / len(train_loader) + nll
        acc = (torch.argmax(logits.mean(0), dim=1) == labels).sum()

        test_report.total += loss.item() / len(train_loader)
        test_report.nll += nll.item() / len(train_loader)
        test_report.log_prior += log_prior.item() / len(train_loader)
        test_report.log_variational_posterior += log_variational_posterior.item() / len(train_loader)
        test_report.acc += acc.item() / len(train_dataset) * 100

        pbar.set_postfix(
            total=test_report.total,
            nll=test_report.nll,
            log_prior=test_report.log_prior,
            log_variational_posterior=test_report.log_variational_posterior,
            acc=test_report.acc,
        )

writer.add_scalar("eval_total",                     test_report.total,                     epoch)
writer.add_scalar("eval_nll",                       test_report.nll,                       epoch)
writer.add_scalar("eval_log_prior",                 test_report.log_prior,                 epoch)
writer.add_scalar("eval_log_variational_posterior", test_report.log_variational_posterior, epoch)
writer.add_scalar("eval_acc",                       test_report.acc,                       epoch)