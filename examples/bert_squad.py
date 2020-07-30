from argparse import ArgumentParser
from bayeformers import to_bayesian
from collections import namedtuple
from examples.hypersearch import HyperSearch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoConfig
from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import SquadV1Processor
from transformers import squad_convert_examples_to_features
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Dict
from typing import Iterable
from typing import Tuple

import bayeformers.nn as bnn
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


RangeTensor = Tuple[torch.Tensor, torch.Tensor]


def f1_train(predicted_range: RangeTensor, target_range: RangeTensor) -> torch.Tensor:
    predicted_start, predicted_end = predicted_range
    target_start,    target_end    = target_range
    
    overlap              = predicted_end - target_start
    overlap[overlap < 0] = 0.0

    precision = overlap / (predicted_end - predicted_start)
    recall    = overlap / (target_end    - target_start)
    f1        = 2 * precision * recall / (precision + recall)
    return f1


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
        self.f1                       : float = 0.0


def dic2cuda(dic: Dict, device: str) -> Dict:
    for key, value in dic.items():
        if isinstance(value, torch.Tensor):
            dic[key] = value.to(device)

    return dic


def setup_model(model_name: str, lower_case: bool) -> Tuple[nn.Module, nn.Module]:
    config    = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=lower_case)
    model     = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    return model, tokenizer


def setup_squadv1_dataset(data_dir: str, tokenizer: nn.Module, test: bool = False, **kwargs) -> Dataset:
    cached_path = os.path.join(data_dir, f"{'dev' if test else 'train'}v1.pth")
    if os.path.isfile(cached_path):
        return torch.load(cached_path)["dataset"] 
    
    processor   = SquadV1Processor()
    fname       = f"{'dev' if test else 'train'}-v1.1.json"
    examples    = processor.get_train_examples(data_dir, fname)
    _, dataset  = squad_convert_examples_to_features(
        examples         = examples,
        tokenizer        = tokenizer,
        is_training      = True,
        return_dataset   = "pt",
        **kwargs
    )

    torch.save({ "dataset": dataset }, cached_path)
    return dataset


def setup_inputs(data: Iterable, model_name: str, model: nn.Module) -> Dict[str, torch.Tensor]:
    inputs = {
        "input_ids"      : data[0],
        "attention_mask" : data[1],
        "token_type_ids" : data[2],
        "start_positions": data[3],
        "end_positions"  : data[4],
    }

    if ("xlm" in model_name) or ("roberta" in model_name) or ("distilbert" in model_name) or ("camembert" in model_name):
        del inputs["token_type_ids"]
    if ("xlnet" in model_name) or ("xlm" in model_name):
        inputs.update({ "cls_index": data[5], "p_mask": data[6] })

    return inputs


def sample_bayesian(
    model: bnn.Model, inputs: Dict[str, torch.Tensor], samples: int, batch_size: int, max_seq_len: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start_logits              = torch.zeros(samples, batch_size, max_seq_len).to(device)
    end_logits                = torch.zeros(samples, batch_size, max_seq_len).to(device)
    log_prior                 = torch.zeros(samples, batch_size             ).to(device)
    log_variational_posterior = torch.zeros(samples, batch_size             ).to(device)

    for sample in range(samples):
        outputs                           = model(**inputs)
        start_logits[sample]              = outputs[1]
        end_logits[sample]                = outputs[2]
        log_prior[sample]                 = model.log_prior()
        log_variational_posterior[sample] = model.log_variational_posterior()

    raw_start_logits          = start_logits
    raw_end_logits            = end_logits
    start_logits              = start_logits.mean(0)
    end_logits                = end_logits.mean(0)
    log_prior                 = log_prior.mean()
    log_variational_posterior = log_variational_posterior.mean()

    return raw_start_logits, raw_end_logits, start_logits, end_logits, log_prior, log_variational_posterior


def train(EXP: str, MODEL_NAME: str, DELTA: float, WEIGHT_DECAY: float, DEVICE: str) -> float:
    EPOCHS           = 5
    BATCH_SIZE       = 8
    SAMPLES          = 10
    FREEZE           = True
    LOGS             = "logs"
    MAX_SEQ_LENGTH   = 384
    DOC_STRIDE       = 128
    MAX_QUERY_LENGTH = 64
    LOWER_CASE       = True
    THREADS          = 4
    LOADER_OPTIONS   = { "num_workers": 6, "pin_memory": True }
    LR               = 5e-5
    ADAM_EPSILON     = 1e-8
    N_WARMUP_STEPS   = 0
    MAX_GRAD_NORM    = 1
    DATA_DIR         = os.path.join("./dataset/squadv1")

    os.makedirs(LOGS, exist_ok=True)
    writer_path = os.path.join(LOGS, f"bayeformers_bert_squad.{EXP}")
    writer_suff = f".DELTA_{DELTA}.WEIGHT_DECAY_{WEIGHT_DECAY}"
    writer      = SummaryWriter(writer_path + writer_suff)

    o_model, tokenizer = setup_model(MODEL_NAME, LOWER_CASE)
    o_model            = o_model.to(DEVICE)

    squadv1       = {
        "max_seq_length"  : MAX_SEQ_LENGTH,
        "doc_stride"      : DOC_STRIDE,
        "max_query_length": MAX_QUERY_LENGTH,
        "threads"         : THREADS
    }
    train_dataset = setup_squadv1_dataset(DATA_DIR, tokenizer=tokenizer, test=False, **squadv1)
    test_dataset  = setup_squadv1_dataset(DATA_DIR, tokenizer=tokenizer, test=True,  **squadv1)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  **LOADER_OPTIONS)
    test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, **LOADER_OPTIONS)

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
            inputs = setup_inputs(inputs, MODEL_NAME, o_model)
            inputs = dic2cuda(inputs, DEVICE)
            
            start_positions = inputs["start_positions"]
            end_positions   = inputs["end_positions"]

            optim.zero_grad()
            
            outputs      = o_model(**inputs)
            start_logits = outputs[1]
            end_logits   = outputs[2]
            
            ignored_idx            = start_logits.size(1)
            start_logits           = start_logits.clamp_(0, ignored_idx)
            end_logits             =   end_logits.clamp_(0, ignored_idx)
            criterion.ignore_index = ignored_idx

            start_loss = criterion(start_logits, start_positions)
            end_loss   = criterion(  end_logits,   end_positions)
            start_acc  = (torch.argmax(start_logits, dim=1) == start_positions).float().sum()
            end_acc    = (torch.argmax(  end_logits, dim=1) ==   end_positions).float().sum()

            loss = 0.5 * (start_loss + end_loss)
            acc  = 0.5 * (start_acc  + end_acc)
            f1   = f1_train(
                (torch.argmax(start_logits, dim=1), torch.argmax(end_logits, dim=1)),
                (start_positions, end_positions)
            ).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(o_model.parameters(), MAX_GRAD_NORM)
            optim.step()

            report.total += loss.item()      / len(train_loader)
            report.acc   += acc.item() * 100 / len(train_dataset)
            report.f1    += f1.item()        / len(train_loader)

            pbar.set_postfix(total=report.total, acc=report.acc, f1=report.f1)

        scheduler.step()
        writer.add_scalar("train_nll", report.total, epoch)
        writer.add_scalar("train_acc", report.acc,   epoch)
        writer.add_scalar("train_f1",  report.f1,    epoch)

        # ============================ TEST =======================================
        o_model.eval()
        report.reset()
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Test")
            for inputs in pbar:
                inputs = setup_inputs(inputs, MODEL_NAME, o_model)
                inputs = dic2cuda(inputs, DEVICE)
                
                start_positions = inputs["start_positions"]
                end_positions   = inputs["end_positions"]

                outputs      = o_model(**inputs)
                start_logits = outputs[1]
                end_logits   = outputs[2]
                
                ignored_idx            = start_logits.size(1)
                start_logits           = start_logits.clamp_(0, ignored_idx)
                end_logits             =   end_logits.clamp_(0, ignored_idx)
                criterion.ignore_index = ignored_idx

                start_loss = criterion(start_logits, start_positions)
                end_loss   = criterion(  end_logits,   end_positions)
                start_acc  = (torch.argmax(start_logits, dim=1) == start_positions).float().sum()
                end_acc    = (torch.argmax(  end_logits, dim=1) ==   end_positions).float().sum()

                loss = 0.5 * (start_loss + end_loss)
                acc  = 0.5 * (start_acc  + end_acc)

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
            inputs = setup_inputs(inputs, MODEL_NAME, o_model)
            inputs = dic2cuda(inputs, DEVICE)

            start_positions = inputs["start_positions"]
            end_positions   = inputs["end_positions"]
            B               = inputs["input_ids"].size(0)

            samples = sample_bayesian(b_model, inputs, SAMPLES, B, MAX_SEQ_LENGTH, DEVICE)
            raw_start_logits, raw_end_logits, start_logits, end_logits, log_prior, log_variational_posterior = samples
            
            ignored_idx            = start_logits.size(1)
            start_logits           = start_logits.clamp_(0, ignored_idx)
            end_logits             =   end_logits.clamp_(0, ignored_idx)
            criterion.ignore_index = ignored_idx

            start_loss    = criterion(start_logits, start_positions)
            end_loss      = criterion(  end_logits,   end_positions)
            start_acc     = (torch.argmax(start_logits, dim=1) == start_positions).float().sum()
            end_acc       = (torch.argmax(  end_logits, dim=1) ==   end_positions).float().sum()
            start_acc_std = np.std([(torch.argmax(start_logits.clamp(0, ignored_idx), dim=1) == start_positions).float().sum().item() for start_logits in raw_start_logits])
            end_acc_std   = np.std([(torch.argmax(  end_logits.clamp(0, ignored_idx), dim=1) ==   end_positions).float().sum().item() for   end_logits in raw_end_logits])

            nll     = 0.5 * (start_loss    + end_loss)
            acc     = 0.5 * (start_acc     + end_acc)
            acc_std = 0.5 * (start_acc_std + end_acc_std)
            loss    = (log_variational_posterior - log_prior) / len(test_loader) + nll

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
            inputs = setup_inputs(inputs, MODEL_NAME, o_model)
            inputs = dic2cuda(inputs, DEVICE)

            start_positions = inputs["start_positions"]
            end_positions   = inputs["end_positions"]
            B               = inputs["input_ids"].size(0)

            optim.zero_grad()

            samples = sample_bayesian(b_model, inputs, SAMPLES, B, MAX_SEQ_LENGTH, DEVICE)
            raw_start_logits, raw_end_logits, start_logits, end_logits, log_prior, log_variational_posterior = samples
            
            ignored_idx            = start_logits.size(1)
            start_logits           = start_logits.clamp_(0, ignored_idx)
            end_logits             =   end_logits.clamp_(0, ignored_idx)
            criterion.ignore_index = ignored_idx

            start_loss    = criterion(start_logits, start_positions)
            end_loss      = criterion(  end_logits,   end_positions)
            start_acc     = (torch.argmax(start_logits, dim=1) == start_positions).float().sum()
            end_acc       = (torch.argmax(  end_logits, dim=1) ==   end_positions).float().sum()
            start_acc_std = np.std([(torch.argmax(start_logits.clamp(0, ignored_idx), dim=1) == start_positions).float().sum().item() for start_logits in raw_start_logits])
            end_acc_std   = np.std([(torch.argmax(  end_logits.clamp(0, ignored_idx), dim=1) ==   end_positions).float().sum().item() for   end_logits in raw_end_logits])

            nll     = 0.5 * (start_loss    + end_loss)
            acc     = 0.5 * (start_acc     + end_acc)
            acc_std = 0.5 * (start_acc_std + end_acc_std)
            loss    = (log_variational_posterior - log_prior) / len(train_loader) + nll
            f1      = f1_train(
                (torch.argmax(start_logits, dim=1), torch.argmax(end_logits, dim=1)),
                (start_positions, end_positions)
            ).mean()

            loss.backward()
            nn.utils.clip_grad_norm_(b_model.parameters(), MAX_GRAD_NORM)
            optim.step()

            report.total                     += loss.item()                      / len(train_loader)
            report.nll                       += nll.item()                       / len(train_loader)
            report.log_prior                 += log_prior.item()                 / len(train_loader)
            report.log_variational_posterior += log_variational_posterior.item() / len(train_loader)
            report.acc                       += acc.item() * 100                 / len(train_dataset)
            report.acc_std                   += acc_std                          / len(train_loader)
            report.f1                        += f1.item()                        / len(train_loader)

            pbar.set_postfix(
                total=report.total,
                nll=report.nll,
                log_prior=report.log_prior,
                log_variational_posterior=report.log_variational_posterior,
                acc=report.acc,
                acc_std=report.acc_std,
                f1=report.f1,
            )

        scheduler.step()
        writer.add_scalar("bayesian_train_nll",     report.nll,     epoch)
        writer.add_scalar("bayesian_train_acc",     report.acc,     epoch)
        writer.add_scalar("bayesian_train_acc_std", report.acc_std, epoch)
        writer.add_scalar("bayesian_train_f1",      report.f1,      epoch)

        # ============================ TEST =======================================
        b_model.eval()
        report.reset()
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Bayesian Test")
            for inputs in pbar:
                inputs = setup_inputs(inputs, MODEL_NAME, o_model)
                inputs = dic2cuda(inputs, DEVICE)

                start_positions = inputs["start_positions"]
                end_positions   = inputs["end_positions"]
                B               = inputs["input_ids"].size(0)

                samples = sample_bayesian(b_model, inputs, SAMPLES, B, MAX_SEQ_LENGTH, DEVICE)
                raw_start_logits, raw_end_logits, start_logits, end_logits, log_prior, log_variational_posterior = samples
            
                ignored_idx            = start_logits.size(1)
                start_logits           = start_logits.clamp_(0, ignored_idx)
                end_logits             =   end_logits.clamp_(0, ignored_idx)
                criterion.ignore_index = ignored_idx

                start_loss    = criterion(start_logits, start_positions)
                end_loss      = criterion(  end_logits,   end_positions)
                start_acc     = (torch.argmax(start_logits, dim=1) == start_positions).float().sum()
                end_acc       = (torch.argmax(  end_logits, dim=1) ==   end_positions).float().sum()
                start_acc_std = np.std([(torch.argmax(start_logits.clamp(0, ignored_idx), dim=1) == start_positions).float().sum().item() for start_logits in raw_start_logits])
                end_acc_std   = np.std([(torch.argmax(  end_logits.clamp(0, ignored_idx), dim=1) ==   end_positions).float().sum().item() for   end_logits in raw_end_logits])

                nll     = 0.5 * (start_loss    + end_loss)
                acc     = 0.5 * (start_acc     + end_acc)
                acc_std = 0.5 * (start_acc_std + end_acc_std)
                loss    = (log_variational_posterior - log_prior) / len(test_loader) + nll

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
    parser.add_argument("--exp",         type=str,   default="exp",                     help="experience name for logs")
    parser.add_argument("--model_name",  type=str,   default="distilbert-base-uncased", help="model name")
    parser.add_argument("--device",      type=str,   default="cuda:0",                  help="device (cpu, cuda:0, ..., cuda:n)")

    args = parser.parse_args()

    hypersearch = HyperSearch()
    hypersearch["DELTA"] = (1e-2, 1e-1)
    hypersearch["WEIGHT_DECAY"] = (1e-3, 0)
    
    score = hypersearch.search(
        train, iterations=10, EXP=args.exp, MODEL_NAME=args.model_name, DEVICE=args.device,
    )
    
    print("=========================== BEST SCORE ===========================")
    print(score)
    print("==================================================================")