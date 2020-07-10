from collections import namedtuple
from bayeformers import to_bayesian

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
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            dic[key] = value.cuda()
    return dic


MODEL_NAME     = "distilbert-base-uncased"
TASK_NAME      = "MRPC"
N_LABELS       = 2
MAX_SEQ_LENGTH = 128
DATA_DIR       = os.path.join("./dataset/glue/data", TASK_NAME)
LOADER_OPTIONS = { "num_workers": 4, "pin_memory": True }
EPOCHS         = 3
BATCH_SIZE     = 8
LR             = 2e-5
WEIGHT_DECAY   = 0
ADAM_EPSILON   = 1e-8
N_WARMUP_STEPS = 0
N_TRAIN_STEPS  = EPOCHS
MAX_GRAD_NORM  = 1
SAMPLES        = 10

config    = AutoConfig.from_pretrained(MODEL_NAME, num_labels=N_LABELS, finetuning_task=TASK_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
model = to_bayesian(model, delta=0.05)
model = model.cuda()

glue = GlueDataTrainingArguments(TASK_NAME, data_dir=DATA_DIR, max_seq_length=MAX_SEQ_LENGTH)

train_dataset = GlueDataset(glue, tokenizer=tokenizer)
test_dataset  = GlueDataset(glue, tokenizer=tokenizer, mode="test")
eval_dataset  = GlueDataset(glue, tokenizer=tokenizer, mode="dev")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate, **LOADER_OPTIONS)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, **LOADER_OPTIONS)
eval_loader  = DataLoader(eval_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, **LOADER_OPTIONS)

params_decay    = [param for name, param in model.named_parameters() if name     in ["bias", "LayerNorm.weight"]]
params_no_decay = [param for name, param in model.named_parameters() if name not in ["bias", "LayerNorm.weight"]]
parameters      = [
    { "params": params_decay,    "weight_decay": WEIGHT_DECAY },
    { "params": params_no_decay, "weight_decay": 0.0 },
]

criterion = nn.CrossEntropyLoss().cuda()
optim     = AdamW(parameters, lr=LR, eps=ADAM_EPSILON)
scheduler = get_linear_schedule_with_warmup(optim, N_WARMUP_STEPS, N_TRAIN_STEPS)

report = Report()
for epoch in tqdm(range(EPOCHS), desc="Epoch"):
    model.train()
    report.reset()
    
    pbar = tqdm(train_loader, desc="Batch")
    for inputs in pbar:
        inputs = dic2cuda(inputs)
        labels = inputs[4]
        optim.zero_grad()

        B = inputs["input_ids"].size(0)
        logits = torch.zeros(SAMPLES, B, N_LABELS).cuda()
        log_prior = torch.zeros(SAMPLES, B).cuda()
        log_variational_posterior = torch.zeros(SAMPLES, B).cuda()

        for sample in range(SAMPLES):
            logits[samples] = model(**inputs)[1]
            log_prior[samples] = model.log_prior()
            log_variational_posterior[samples] = model.log_variational_posterior()

        nll = criterion(logits.mean(0), labels, reduction="sum")
        log_prior = log_prior.mean()
        log_variational_posterior = log_variational_posterior.mean()
        acc = (torch.argmax(logits.mean(0), dim=1) == labels).sum()

        loss = (log_variational_posterior - log_prior) / len(train_loader) + nll
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), MAX_GRAD_NORM)

        optim.step()
        scheduler.step()

        report.total += loss.item() / len(train_loader)
        report.nll += nll.item() / len(train_loader)
        report.log_prior += log_prior.item() / len(train_loader)
        report.log_variational_posterior += log_variational_posterior.item() / len(train_loader)
        report.acc += acc.item() / len(train_dataset) * 100

        pbar.set_postfix(
            total=report.total,
            nll=report.nll,
            log_prior=report.log_prior,
            log_variational_posterior=report.log_variational_posterior,
            acc=report.acc,
        )