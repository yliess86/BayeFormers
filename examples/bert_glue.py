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

import os
import bayeformers.nn as bnn
import torch
import torch.nn as nn
import torch.nn.functional as F


def train(EXP: str, MODEL_NAME: str, TASK_NAME: str, N_LABELS: int, DELTA: float, WEIGHT_DECAY: float, DEVICE: str) -> float:
    EPOCHS         = 1 # 5
    BATCH_SIZE     = 8
    SAMPLES        = 2 # 10
    FREEZE         = True
    LOGS           = "logs"
    MAX_SEQ_LENGTH = 128
    LOADER_OPTIONS = { "num_workers": 6, "pin_memory": True }
    LR             = 2e-5
    ADAM_EPSILON   = 1e-8
    N_WARMUP_STEPS = 0
    MAX_GRAD_NORM  = 1
    DATA_DIR       = os.path.join("./dataset/glue/data", TASK_NAME)


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
    writer = SummaryWriter(
        os.path.join(LOGS, f"bayeformers_bert_glue.{EXP}"),
        filename_suffix=f".DELTA_{DELTA}.WEIGHT_DECAY_{WEIGHT_DECAY}"
    )

    config    = AutoConfig.from_pretrained(MODEL_NAME, num_labels=N_LABELS, finetuning_task=TASK_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    o_model   = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    o_model   = o_model.to(DEVICE)

    glue = GlueDataTrainingArguments(TASK_NAME, data_dir=DATA_DIR, max_seq_length=MAX_SEQ_LENGTH)

    train_dataset = GlueDataset(glue, tokenizer=tokenizer)
    test_dataset  = GlueDataset(glue, tokenizer=tokenizer, mode="dev")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate, **LOADER_OPTIONS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, **LOADER_OPTIONS)

    params_decay    = [param for name, param in o_model.named_parameters() if name     in ["bias", "LayerNorm.weight"]]
    params_no_decay = [param for name, param in o_model.named_parameters() if name not in ["bias", "LayerNorm.weight"]]
    parameters      = [
        { "params": params_decay,    "weight_decay": WEIGHT_DECAY },
        { "params": params_no_decay, "weight_decay": 0.0 },
    ]

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
            inputs = dic2cuda(inputs)
            labels = inputs["labels"]
            optim.zero_grad()

            logits = o_model(**inputs)[1]
            loss = criterion(logits.view(-1, N_LABELS), labels.view(-1))
            acc = (torch.argmax(logits, dim=1) == labels).float().sum()

            loss.backward()
            nn.utils.clip_grad_norm_(o_model.parameters(), MAX_GRAD_NORM)
            optim.step()

            report.total += loss.item() / len(train_loader)
            report.acc += acc.item() / len(train_dataset) * 100

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
                inputs = dic2cuda(inputs)
                labels = inputs["labels"]

                logits = o_model(**inputs)[1]
                loss = criterion(logits.view(-1, N_LABELS), labels.view(-1))
                acc = (torch.argmax(logits, dim=1) == labels).float().sum()

                report.total += loss.item() / len(test_loader)
                report.acc += acc.item() / len(test_dataset) * 100

                pbar.set_postfix(total=report.total, acc=report.acc)

        writer.add_scalar("test_nll", report.total, epoch)
        writer.add_scalar("test_acc", report.acc,   epoch)

    # ============================ EVALUTATION ====================================
    b_model                  = to_bayesian(o_model, delta=DELTA, freeze=FREEZE)
    b_model.model.classifier = bnn.Linear.from_frequentist(o_model.classifier)
    b_model                  = b_model.to(DEVICE)

    b_model.eval()
    report.reset()

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Bayesian Eval")
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

            nll = criterion(logits.mean(0).view(-1, N_LABELS), labels.view(-1))
            log_prior = log_prior.mean()
            log_variational_posterior = log_variational_posterior.mean()
            acc = (torch.argmax(logits.mean(0), dim=1) == labels).float().sum()
            loss = (log_variational_posterior - log_prior) / len(test_loader) + nll

            report.total += loss.item() / len(test_loader)
            report.nll += nll.item() / len(test_loader)
            report.log_prior += log_prior.item() / len(test_loader)
            report.log_variational_posterior += log_variational_posterior.item() / len(test_loader)
            report.acc += acc.item() / len(test_dataset) * 100

            pbar.set_postfix(
                total=report.total,
                nll=report.nll,
                log_prior=report.log_prior,
                log_variational_posterior=report.log_variational_posterior,
                acc=report.acc,
            )

    writer.add_scalar("bayesian_eval_nll", report.nll, epoch)
    writer.add_scalar("bayesian_eval_acc", report.acc, epoch)

    params_decay    = [param for name, param in b_model.named_parameters() if name     in ["bias", "LayerNorm.weight"]]
    params_no_decay = [param for name, param in b_model.named_parameters() if name not in ["bias", "LayerNorm.weight"]]
    parameters      = [
        { "params": params_decay,    "weight_decay": WEIGHT_DECAY },
        { "params": params_no_decay, "weight_decay": 0.0 },
    ]

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optim     = AdamW(parameters, lr=LR, eps=ADAM_EPSILON)
    scheduler = get_linear_schedule_with_warmup(optim, N_WARMUP_STEPS, EPOCHS)

    for epoch in tqdm(range(EPOCHS), desc="Bayesian Epoch"):

        # ============================ TRAIN ======================================
        b_model.train()
        report.reset()
        
        pbar = tqdm(train_loader, desc="Bayesian Train")
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

            nll = criterion(logits.mean(0).view(-1, N_LABELS), labels.view(-1))
            log_prior = log_prior.mean()
            log_variational_posterior = log_variational_posterior.mean()
            loss = (log_variational_posterior - log_prior) / len(train_loader) + nll
            acc = (torch.argmax(logits.mean(0), dim=1) == labels).float().sum()

            loss.backward()
            nn.utils.clip_grad_norm_(b_model.parameters(), MAX_GRAD_NORM)
            optim.step()

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

        scheduler.step()
        writer.add_scalar("bayesian_train_nll", report.nll, epoch)
        writer.add_scalar("bayesian_train_acc", report.acc, epoch)

        # ============================ TEST =======================================
        b_model.eval()
        report.reset()
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Bayesian Test")
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

                nll = criterion(logits.mean(0).view(-1, N_LABELS), labels.view(-1))
                log_prior = log_prior.mean()
                log_variational_posterior = log_variational_posterior.mean()
                loss = (log_variational_posterior - log_prior) / len(test_loader) + nll
                acc = (torch.argmax(logits.mean(0), dim=1) == labels).float().sum()

                report.total += loss.item() / len(test_loader)
                report.nll += nll.item() / len(test_loader)
                report.log_prior += log_prior.item() / len(test_loader)
                report.log_variational_posterior += log_variational_posterior.item() / len(test_loader)
                report.acc += acc.item() / len(test_dataset) * 100

                pbar.set_postfix(
                    total=report.total,
                    nll=report.nll,
                    log_prior=report.log_prior,
                    log_variational_posterior=report.log_variational_posterior,
                    acc=report.acc,
                )

        writer.add_scalar("bayesian_test_nll", report.nll, epoch)
        writer.add_scalar("bayesian_test_acc", report.acc, epoch)

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
    hypersearch["DELTA"] = (1e-6, 1e-1)
    hypersearch["WEIGHT_DECAY"] = (1e-6, 1e-1)
    
    score = hypersearch.search(
        train, iterations=10,
        EXP=args.exp, MODEL_NAME=args.model_name, TASK_NAME=args.task_name,
        N_LABELS=args.n_labels, DEVICE=args.device,
    )
    
    print("=========================== BEST SCORE ===========================")
    print(score)
    print("==================================================================")