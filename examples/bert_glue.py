from argparse import ArgumentParser
from bayeformers import to_bayesian
from collections import namedtuple
from examples.hypersearch import HyperSearch
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
from typing import Tuple

import bayeformers.nn as bnn
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Report:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total                    : float = 0.0
        self.acc                      : float = 0.0
        self.acc_std                  : float = 0.0
        self.nll                      : float = 0.0
        self.log_prior                : float = 0.0
        self.log_variational_posterior: float = 0.0


def dic2cuda(dic: Dict, device: str) -> Dict:
    for key, value in dic.items():
        if isinstance(value, torch.Tensor):
            dic[key] = value.to(device)

    return dic


def setup_model(model_name: str, task_name: str, n_labels: int) -> Tuple[nn.Module, nn.Module] :
    config    = AutoConfig.from_pretrained(model_name, num_labels=n_labels, finetuning_task=task_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    return model, tokenizer


def sample_bayesian(
    model: bnn.Model, inputs: Dict[str, torch.Tensor], samples: int, batch_size: int, n_labels: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits                    = torch.zeros(samples, batch_size, n_labels).to(device)
    log_prior                 = torch.zeros(samples, batch_size          ).to(device)
    log_variational_posterior = torch.zeros(samples, batch_size          ).to(device)

    for sample in range(samples):
        logits[sample]                    = model(**inputs)[1]
        log_prior[sample]                 = model.log_prior()
        log_variational_posterior[sample] = model.log_variational_posterior()

    raw_logits                = logits
    logits                    = logits.mean(0).view(-1, n_labels)
    log_prior                 = log_prior.mean()
    log_variational_posterior = log_variational_posterior.mean()

    return raw_logits, logits, log_prior, log_variational_posterior


def train(EXP: str, MODEL_NAME: str, TASK_NAME: str, N_LABELS: int, DELTA: float, WEIGHT_DECAY: float, DEVICE: str) -> float:
    EPOCHS         = 5
    BATCH_SIZE     = 8
    SAMPLES        = 10
    FREEZE         = True
    LOGS           = "logs"
    MAX_SEQ_LENGTH = 128
    LOADER_OPTIONS = { "num_workers": 6, "pin_memory": True }
    LR             = 2e-5
    ADAM_EPSILON   = 1e-8
    N_WARMUP_STEPS = 0
    MAX_GRAD_NORM  = 1
    DATA_DIR       = os.path.join("./dataset/glue/data", TASK_NAME)

    os.makedirs(LOGS, exist_ok=True)
    writer_path = os.path.join(LOGS, f"bayeformers_bert_glue.{EXP}")
    writer_suff = f".DELTA_{DELTA}.WEIGHT_DECAY_{WEIGHT_DECAY}"
    writer      = SummaryWriter(writer_path + writer_suff)
    
    o_model, tokenizer = setup_model(MODEL_NAME, TASK_NAME, N_LABELS)
    o_model            = o_model.to(DEVICE)

    glue          = GlueDataTrainingArguments(TASK_NAME, data_dir=DATA_DIR, max_seq_length=MAX_SEQ_LENGTH)
    train_dataset = GlueDataset(glue, tokenizer=tokenizer)
    test_dataset  = GlueDataset(glue, tokenizer=tokenizer, mode="dev")
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate, **LOADER_OPTIONS)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, **LOADER_OPTIONS)
    
    decay           = [param for name, param in o_model.named_parameters() if name     in ["bias", "LayerNorm.weight"]]
    no_decay        = [param for name, param in o_model.named_parameters() if name not in ["bias", "LayerNorm.weight"]]
    params_decay    = { "params": decay,    "weight_decay": WEIGHT_DECAY }
    params_no_decay = { "params": no_decay, "weight_decay": 0.0 }
    parameters      = [params_decay, params_no_decay]

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optim     = AdamW(parameters, lr=LR, eps=ADAM_EPSILON)
    scheduler = get_linear_schedule_with_warmup(optim, N_WARMUP_STEPS, EPOCHS)

    report = Report()
    for epoch in tqdm(range(EPOCHS), desc="Epoch"):

        # ============================ TRAIN ======================================
        o_model.train()
        report.reset()
        
        pbar = tqdm(train_loader, desc="Train")
        for inputs in pbar:
            inputs = dic2cuda(inputs, DEVICE)
            labels = inputs["labels"]

            optim.zero_grad()
            logits = o_model(**inputs)[1]
            loss   = criterion(logits.view(-1, N_LABELS), labels.view(-1))
            acc    = (torch.argmax(logits, dim=1) == labels).float().sum()

            loss.backward()
            nn.utils.clip_grad_norm_(o_model.parameters(), MAX_GRAD_NORM)
            optim.step()

            report.total += loss.item()      / len(train_loader)
            report.acc   += acc.item() * 100 / len(train_dataset)

            pbar.set_postfix(total=report.total, acc=report.acc)

        scheduler.step()
        writer.add_scalar("train_nll", report.total, epoch)
        writer.add_scalar("train_acc", report.acc,   epoch)

        # ============================ TEST =======================================
        o_model.eval()
        report.reset()
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Test")
            for inputs in pbar:
                inputs = dic2cuda(inputs, DEVICE)
                labels = inputs["labels"]

                logits = o_model(**inputs)[1]
                loss   = criterion(logits.view(-1, N_LABELS), labels.view(-1))
                acc    = (torch.argmax(logits, dim=1) == labels).float().sum()

                report.total += loss.item()       / len(test_loader)
                report.acc   += acc.item() * 100  / len(test_dataset)

                pbar.set_postfix(total=report.total, acc=report.acc)

        writer.add_scalar("test_nll", report.total, epoch)
        writer.add_scalar("test_acc", report.acc,   epoch)

    # ============================ EVALUTATION ====================================
    b_model                  = to_bayesian(o_model, delta=DELTA, freeze=FREEZE)
    b_model                  = b_model.to(DEVICE)

    b_model.eval()
    report.reset()

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Bayesian Eval")
        for inputs in pbar:
            inputs = dic2cuda(inputs, DEVICE)
            labels = inputs["labels"]
            B      = inputs["input_ids"].size(0)

            samples = sample_bayesian(b_model, inputs, SAMPLES, B, N_LABELS, DEVICE)
            raw_logits, logits, log_prior, log_variational_posterior = samples

            nll     = criterion(logits, labels.view(-1))            
            loss    = (log_variational_posterior - log_prior) / len(test_loader) + nll
            acc     = (torch.argmax(logits, dim=1) == labels).float().sum()
            acc_std = np.std([(torch.argmax(logits, dim=1) == labels).float().sum().item() for logits in raw_logits])

            report.total                     += loss.item()                      / len(test_loader)
            report.nll                       += nll.item()                       / len(test_loader)
            report.log_prior                 += log_prior.item()                 / len(test_loader)
            report.log_variational_posterior += log_variational_posterior.item() / len(test_loader)
            report.acc                       += acc.item() * 100                 / len(test_dataset)
            report.acc_std                   += acc_std                          / len(test_loader)

            pbar.set_postfix(
                total=report.total,
                nll=report.nll,
                log_prior=report.log_prior,
                log_variational_posterior=report.log_variational_posterior,
                acc=report.acc,
                acc_std=report.acc_std,
            )

    writer.add_scalar("bayesian_eval_nll",     report.nll,     epoch)
    writer.add_scalar("bayesian_eval_acc",     report.acc,     epoch)
    writer.add_scalar("bayesian_eval_acc_std", report.acc_std, epoch)

    decay           = [param for name, param in b_model.named_parameters() if name     in ["bias", "LayerNorm.weight"]]
    no_decay        = [param for name, param in b_model.named_parameters() if name not in ["bias", "LayerNorm.weight"]]
    params_decay    = { "params": decay,    "weight_decay": WEIGHT_DECAY }
    params_no_decay = { "params": no_decay, "weight_decay": 0.0 }
    parameters      = [params_decay, params_no_decay]

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optim     = AdamW(parameters, lr=LR, eps=ADAM_EPSILON)
    scheduler = get_linear_schedule_with_warmup(optim, N_WARMUP_STEPS, EPOCHS)

    for epoch in tqdm(range(EPOCHS), desc="Bayesian Epoch"):

        # ============================ TRAIN ======================================
        b_model.train()
        report.reset()
        
        pbar = tqdm(train_loader, desc="Bayesian Train")
        for inputs in pbar:
            inputs = dic2cuda(inputs, DEVICE)
            labels = inputs["labels"]
            B      = inputs["input_ids"].size(0)

            optim.zero_grad()
            samples = sample_bayesian(b_model, inputs, SAMPLES, B, N_LABELS, DEVICE)
            raw_logits, logits, log_prior, log_variational_posterior = samples

            nll     = criterion(logits, labels.view(-1))            
            loss    = (log_variational_posterior - log_prior) / len(train_loader) + nll
            acc     = (torch.argmax(logits, dim=1) == labels).float().sum()
            acc_std = np.std([(torch.argmax(logits, dim=1) == labels).float().sum().item() for logits in raw_logits])

            loss.backward()
            nn.utils.clip_grad_norm_(b_model.parameters(), MAX_GRAD_NORM)
            optim.step()

            report.total                     += loss.item()                      / len(train_loader)
            report.nll                       += nll.item()                       / len(train_loader)
            report.log_prior                 += log_prior.item()                 / len(train_loader)
            report.log_variational_posterior += log_variational_posterior.item() / len(train_loader)
            report.acc                       += acc.item() * 100                 / len(train_dataset)
            report.acc_std                   += acc_std                          / len(train_loader)

            pbar.set_postfix(
                total=report.total,
                nll=report.nll,
                log_prior=report.log_prior,
                log_variational_posterior=report.log_variational_posterior,
                acc=report.acc,
                acc_std=acc_std,
            )

        scheduler.step()
        writer.add_scalar("bayesian_train_nll",     report.nll,     epoch)
        writer.add_scalar("bayesian_train_acc",     report.acc,     epoch)
        writer.add_scalar("bayesian_train_acc_std", report.acc_std, epoch)

        # ============================ TEST =======================================
        b_model.eval()
        report.reset()
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Bayesian Test")
            for inputs in pbar:
                inputs = dic2cuda(inputs, DEVICE)
                labels = inputs["labels"]
                B      = inputs["input_ids"].size(0)

                samples = sample_bayesian(b_model, inputs, SAMPLES, B, N_LABELS, DEVICE)
                raw_logits, logits, log_prior, log_variational_posterior = samples

                nll     = criterion(logits, labels.view(-1))
                loss    = (log_variational_posterior - log_prior) / len(test_loader) + nll
                acc     = (torch.argmax(logits, dim=1) == labels).float().sum()
                acc_std = np.std([(torch.argmax(logits, dim=1) == labels).float().sum().item() for logits in raw_logits])

                report.total                     += loss.item()                      / len(test_loader)
                report.nll                       += nll.item()                       / len(test_loader)
                report.log_prior                 += log_prior.item()                 / len(test_loader)
                report.log_variational_posterior += log_variational_posterior.item() / len(test_loader)
                report.acc                       += acc.item() * 100                 / len(test_dataset)
                report.acc_std                   += acc_std                          / len(test_loader)

                pbar.set_postfix(
                    total=report.total,
                    nll=report.nll,
                    log_prior=report.log_prior,
                    log_variational_posterior=report.log_variational_posterior,
                    acc=report.acc,
                    acc_std=report.acc_std,
                )

        writer.add_scalar("bayesian_test_nll",     report.nll,     epoch)
        writer.add_scalar("bayesian_test_acc",     report.acc,     epoch)
        writer.add_scalar("bayesian_test_acc_std", report.acc_std, epoch)

    torch.save({
        "weight_decay": WEIGHT_DECAY,
        "delta"       : DELTA,
        "acc"         : report.acc,
        "acc_std"     : report.acc_std,
        "model"       : b_model.state_dict()
    }, f"{writer_path + writer_suff}.pth")

    return report.acc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp",           type=str,   default="exp",                     help="experience name for logs")
    parser.add_argument("--model_name",    type=str,   default="distilbert-base-uncased", help="model name")
    parser.add_argument("--task_name",     type=str,   default="MRPC",                    help="task name")
    parser.add_argument("--n_labels",      type=int,   default=2,                         help="number of classes")
    parser.add_argument("--device",        type=str,   default="cuda:0",                  help="device (cpu, cuda:0, ..., cuda:n)")

    args = parser.parse_args()

    hypersearch = HyperSearch()
    hypersearch["DELTA"] = ((1e-2, 1e-1), True)
    hypersearch["WEIGHT_DECAY"] = ((1e-3, 0), False)
    
    score = hypersearch.search(
        train, iterations=10,
        EXP=args.exp, MODEL_NAME=args.model_name, TASK_NAME=args.task_name,
        N_LABELS=args.n_labels, DEVICE=args.device,
    )
    
    print("=========================== BEST SCORE ===========================")
    print(score)
    print("==================================================================")