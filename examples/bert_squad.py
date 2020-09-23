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
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.metrics.squad_metrics import squad_evaluate
from transformers.data.processors.squad import SquadResult
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple
from torch import Tensor
import random
import json

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
        self.em                       : float = 0.0
        self.f1                       : float = 0.0

class Section:
    def __init__(self, parent: "Section", name : str = None):
        self.parent = parent
        self.data   = {}

    def __setitem__(self, key, value):
        # if self.not_leaf:
        #     raise Exception("Cannot store in non-leaf node")
        self.data[key] = value

    def is_root(self):
        return self.parent == None

    def sub_section(self, name = None) -> "Section":
        if not name:
            if len(self.data) == 0:
                self.data = []
            else:
                raise Exception("Cannot make a non empty node a data node")
            sub = Section(self) # Data node
            self.data.append(sub)
            return sub
        if name in self.data:
            raise Exception("Got same section name, this would erase data")
        s = Section(self)
        self.data[name] = s
        return s

    def __repr__(self):
        return repr(self.data)


class Dumper:
    
    def __init__(self, filename: str = None) -> None:
        if not finename:
            filename = f'unnamed.dump'
        self.original_filename : str = filename
        self.filename : str = filename
        if os.exists(self.filename):
            raise Exception("Dump file already exists")
        if not os.access(self.filename, os.W_OK):
            raise Exception("Access denied to create file")

    def reset(self):
        self.root_section    = Section(None)
        self.current_section = self.root_section

    def __call__(self, name: str = None, value = None):
        section_name = name
        if name and value:
            section_name += f'={value}'
        self.current_section = self.current_section.sub_section(section_name)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.current_section = self.current_section.parent
        if self.current_section.is_root():
            self.dump()

    def dump(self):
        posfix = "".join([random.choice(string.ascii_letters + string.digits) for n in range(5)]).upper()
        self.filename = f'{self.original_filename}.{self.section_name}.{postfix}'
        print(f'Dumping results to {self.filename}')
        with open(self.filename, 'w+') as fh:
            fh.write(repr(self.root_section))
        print("Done dumping results")
        self.reset()

    def __setitem__(self, name: str, value):
        if type(value) == torch.Tensor:
            value = value.tolist()
        self.current_section[name] = value

    
def to_list(tensor: torch.Tensor) -> List[torch.Tensor]:
    return tensor.detach().cpu().tolist()


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


def setup_squadv1_dataset(data_dir: str, tokenizer: nn.Module, test: bool = False, **kwargs) -> Tuple[Dataset, torch.Tensor, torch.Tensor]:
    cached_path = os.path.join(data_dir, f"{'dev' if test else 'train'}v1.pth")
    if os.path.isfile(cached_path):
        ckpt = torch.load(cached_path)
        return ckpt["dataset"], ckpt["examples"], ckpt["features"]
    
    processor   = SquadV1Processor()
    fname       = f"{'dev' if test else 'train'}-v1.1.json"
    getter      = processor.get_dev_examples if test else processor.get_train_examples
    examples    = getter(data_dir, fname)
    features, dataset  = squad_convert_examples_to_features(
        examples         = examples,
        tokenizer        = tokenizer,
        is_training      = not test,
        return_dataset   = "pt",
        **kwargs
    )

    torch.save({ "dataset": dataset, "examples": examples, "features": features }, cached_path)
    return dataset, examples, features


def setup_inputs(data: Iterable, model_name: str, model: nn.Module, test: bool = False) -> Dict[str, torch.Tensor]:
    inputs = {
        "input_ids"      : data[0],
        "attention_mask" : data[1],
        "token_type_ids" : data[2],
        "start_positions": data[3] if not test else None,
        "end_positions"  : data[4] if not test else None,
        "feature_indices": None if not test else data[3],
    }

    if test:
        del inputs["start_positions"]
        del inputs["end_positions"]
    else:
        del inputs["feature_indices"]

    if ("xlm" in model_name) or ("roberta" in model_name) or ("distilbert" in model_name) or ("camembert" in model_name):
        del inputs["token_type_ids"]

    return inputs


def sample_bayesian(
    model: bnn.Model, inputs: Dict[str, torch.Tensor], samples: int, batch_size: int, max_seq_len: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    start_logits               : torch.Tensor     = torch.zeros(samples, batch_size, max_seq_len).to(device)
    end_logits                 : torch.Tensor     = torch.zeros(samples, batch_size, max_seq_len).to(device)
    log_prior                  : torch.Tensor     = torch.zeros(samples, batch_size             ).to(device)
    log_variational_posterior  : torch.Tensor     = torch.zeros(samples, batch_size             ).to(device)
        
    for sample in range(samples):
        outputs                           = model(**inputs)
        start_logits[sample]              = outputs[-2]
        end_logits[sample]                = outputs[-1]
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
    EPOCHS            = 3
    BATCH_SIZE        = 13
    SAMPLES           = 10
    FREEZE            = True
    LOGS              = "logs"
    DOC_STRIDE        = 128
    MAX_SEQ_LENGTH    = 384
    MAX_QUERY_LENGTH  = 64
    MAX_ANSWER_LENGTH = 30
    N_BEST_SIZE       = 20
    NULL_SCORE_THRESH = 0.0
    LOWER_CASE        = True
    THREADS           = 4
    LOADER_OPTIONS    = { "num_workers": 10, "pin_memory": True }
    LR                = 5e-5
    ADAM_EPSILON      = 1e-8
    N_WARMUP_STEPS    = 0
    MAX_GRAD_NORM     = 1
    DATA_DIR          = os.path.join("./dataset/squadv1")

    dumper = Dumper(f'dumps/dump_{EXP}_{MODEL_NAME}_{DELTA}.dump')

    os.makedirs(LOGS, exist_ok=True)
    writer_name = f"bayeformers_bert_squad.{EXP}"
    writer_path = os.path.join(LOGS, writer_name)
    writer_suff = f".DELTA_{DELTA}.WEIGHT_DECAY_{WEIGHT_DECAY}"
    writer      = SummaryWriter(writer_path + writer_suff)

    o_model, tokenizer = setup_model(MODEL_NAME, LOWER_CASE)
    o_model = torch.nn.DataParallel(o_model, device_ids=[0, 1, 2, 3])
    o_model.to(DEVICE)

    squadv1 = {
        "max_seq_length"  : MAX_SEQ_LENGTH,
        "doc_stride"      : DOC_STRIDE,
        "max_query_length": MAX_QUERY_LENGTH,
        "threads"         : THREADS
    }
    
    train_dataset, train_examples, train_features = setup_squadv1_dataset(DATA_DIR, tokenizer=tokenizer, test=False, **squadv1)
    test_dataset,  test_examples,  test_features  = setup_squadv1_dataset(DATA_DIR, tokenizer=tokenizer, test=True,  **squadv1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  **LOADER_OPTIONS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, **LOADER_OPTIONS)

    decay           = [param for name, param in o_model.named_parameters() if name     in ["bias", "LayerNorm.weight"]]
    no_decay        = [param for name, param in o_model.named_parameters() if name not in ["bias", "LayerNorm.weight"]]
    params_decay    = { "params": decay,    "weight_decay": WEIGHT_DECAY }
    params_no_decay = { "params": no_decay, "weight_decay": 0.0 }
    parameters      = [params_decay, params_no_decay]

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optim     = AdamW(parameters, lr=LR, eps=ADAM_EPSILON)
    scheduler = get_linear_schedule_with_warmup(optim, N_WARMUP_STEPS, EPOCHS)

    # =========================== FREQUENTIST ==================================
    
    report = Report()
    with dumper("frequentist_train"):
        for epoch in tqdm(range(EPOCHS), desc="Epoch"):

            # ============================ TRAIN ======================================
            o_model.train()
            report.reset()

            with dumper("epoch", epoch):
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
                    end_logits             = end_logits.clamp_(0, ignored_idx)
                    criterion.ignore_index = ignored_idx

                    with dumper():
                        dumper['start_positions'] = start_positions
                        dumper['end_positions']   = end_positions
                        dumper['start_logits']    = start_logits
                        dumper['end_logits']      = end_logits

                    start_loss = criterion(start_logits, start_positions)
                    end_loss   = criterion(  end_logits,   end_positions)
                    start_acc  = (torch.argmax(start_logits, dim=1) == start_positions).float().sum()
                    end_acc    = (torch.argmax(  end_logits, dim=1) ==   end_positions).float().sum()

                    loss = 0.5 * (start_loss + end_loss)
                    acc  = 0.5 * (start_acc  + end_acc)

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
        
        with dumper.section("frequentist_test"):
            with torch.no_grad():
                results = []
                pbar    = tqdm(test_loader, desc="Test")
                for inputs in pbar:
                    inputs          = setup_inputs(inputs, MODEL_NAME, o_model, True)
                    inputs          = dic2cuda(inputs, DEVICE)
                    feature_indices = inputs["feature_indices"]
                    
                    del inputs["feature_indices"]
                    outputs = o_model(**inputs)

                    for i, feature_idx in enumerate(feature_indices):
                        eval_feature             = test_features[feature_idx.item()]
                        unique_id                = int(eval_feature.unique_id)
                        output                   = [to_list(output[i]) for output in outputs]
                        start_logits, end_logits = output
                        result                   = SquadResult(unique_id, start_logits, end_logits)
                        results.append(result)
                        
                        with dumper():
                            dumper['unique_id']     = unique_id
                            dumper['start_logits']  = start_logits
                            dumper['end_logits']    = end_logits

                predictions = compute_predictions_logits(
                    test_examples, test_features, results,
                    N_BEST_SIZE, MAX_ANSWER_LENGTH, LOWER_CASE,
                    os.path.join(LOGS, f"preds.frequentist.test.{writer_name + writer_suff}.json"),
                    os.path.join(LOGS, f"nbestpreds.frequentist.test.{writer_name + writer_suff}.json"),
                    None, True, False, NULL_SCORE_THRESH, tokenizer,
                )

            results      = squad_evaluate(test_examples, predictions)
            report.em    = results["exact"]
            report.f1    = results["f1"]
            report.total = results["total"]
            
            print(f'em={report.em}, f1={report.f1}, total={report.total}')
            writer.add_scalar("test_em",    report.em,    epoch)
            writer.add_scalar("test_f1",    report.f1,    epoch)
            writer.add_scalar("test_total", report.total, epoch)

    # ============================ EVALUTATION ====================================
    b_model = to_bayesian(o_model, delta=DELTA, freeze=FREEZE)
    b_model = b_model.to(DEVICE)

    b_model.eval()
    report.reset()

    with dumper("bayesian_eval_before_train"):
        with torch.no_grad():
            results = []
            pbar    = tqdm(test_loader, desc="Bayesian Eval")
            for inputs in pbar:
                inputs          = setup_inputs(inputs, MODEL_NAME, o_model, True)
                inputs          = dic2cuda(inputs, DEVICE)
                feature_indices = inputs["feature_indices"]
                B               = inputs["input_ids"].size(0)

                del inputs["feature_indices"]
                samples = sample_bayesian(b_model, inputs, SAMPLES, B, MAX_SEQ_LENGTH, DEVICE)
                _, _, start_logits, end_logits, log_prior, log_variational_posterior = samples
                
                start_logits_list = start_logits.tolist()
                end_logits_list   = end_logits.tolist()

                for i, feature_idx in enumerate(feature_indices):
                    eval_feature = test_features[feature_idx.item()]
                    unique_id    = int(eval_feature.unique_id)
                    result       = SquadResult(unique_id, start_logits_list[i], end_logits_list[i])
                    results.append(result)

                    with dumper():
                        dumper['unique_id']     = unique_id
                        dumper['start_logits']  = start_logits_list[i]
                        dumper['end_logits']    = end_logits_list[i]

            predictions = compute_predictions_logits(
                test_examples, test_features, results,
                N_BEST_SIZE, MAX_ANSWER_LENGTH, LOWER_CASE,
                os.path.join(LOGS, f"preds.bayesian.eval.{writer_name + writer_suff}.json"),
                os.path.join(LOGS, f"nbestpreds.bayesian.eval.{writer_name + writer_suff}.json"),
                None, True, False, NULL_SCORE_THRESH, tokenizer,
            )

            results      = squad_evaluate(test_examples, predictions)
            report.em    = results["exact"]
            report.f1    = results["f1"]
            report.total = results["total"]
            
            print(f'em={report.em}, f1={report.f1}, total={report.total}')
            writer.add_scalar("bayesian_eval_em",    report.em,    epoch)
            writer.add_scalar("bayesian_eval_f1",    report.f1,    epoch)
            writer.add_scalar("bayesian_eval_total", report.total, epoch)

    # ============================ BAYESIAN ======================================

    decay           = [param for name, param in b_model.named_parameters() if name     in ["bias", "LayerNorm.weight"]]
    no_decay        = [param for name, param in b_model.named_parameters() if name not in ["bias", "LayerNorm.weight"]]
    params_decay    = { "params": decay,    "weight_decay": WEIGHT_DECAY }
    params_no_decay = { "params": no_decay, "weight_decay": 0.0 }
    parameters      = [params_decay, params_no_decay]

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optim     = AdamW(parameters, lr=LR, eps=ADAM_EPSILON)
    scheduler = get_linear_schedule_with_warmup(optim, N_WARMUP_STEPS, EPOCHS)

    with dumper("bayesian_train"):
        for epoch in tqdm(range(EPOCHS), desc="Bayesian Epoch"):
            with dumper("epoch", epoch):
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

                    with dumper():
                        dumper['start_positions']           = start_positions
                        dumper['end_positions']             = end_positions
                        dumper['start_logits']              = start_logits
                        dumper['end_logits']                = end_logits
                        dumper['log_prior']                 = log_prior
                        dumper['log_variational_posterior'] = log_variational_posterior

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
                        acc_std=report.acc_std,
                    )

                scheduler.step()
                writer.add_scalar("bayesian_train_nll",     report.nll,     epoch)
                writer.add_scalar("bayesian_train_acc",     report.acc,     epoch)
                writer.add_scalar("bayesian_train_acc_std", report.acc_std, epoch)

    # ============================ TEST =======================================
    b_model.eval()
    report.reset()
    
    with dumper("bayesian_test_after_train"):
        with torch.no_grad():
            results = []
            pbar    = tqdm(test_loader, desc="Bayesian Test")
            for inputs in pbar:
                inputs          = setup_inputs(inputs, MODEL_NAME, o_model, True)
                inputs          = dic2cuda(inputs, DEVICE)
                feature_indices = inputs["feature_indices"]
                B               = inputs["input_ids"].size(0)

                del inputs["feature_indices"]
                samples = sample_bayesian(b_model, inputs, SAMPLES, B, MAX_SEQ_LENGTH, DEVICE)
                _, _, start_logits, end_logits, log_prior, log_variational_posterior = samples
                
                start_logits_list   = start_logits.tolist()
                end_logits_list     = end_logits.tolist()

                for i, feature_idx in enumerate(feature_indices):
                    eval_feature = test_features[feature_idx.item()]
                    unique_id    = int(eval_feature.unique_id)
                    result       = SquadResult(unique_id, start_logits_list[i], end_logits_list[i])
                    results.append(result)

                    with dumper():
                        dumper['unique_id']     = unique_id
                        dumper['start_logits']  = start_logits_list[i]
                        dumper['end_logits']    = end_logits_list[i]

            predictions = compute_predictions_logits(
                test_examples, test_features, results,
                N_BEST_SIZE, MAX_ANSWER_LENGTH, LOWER_CASE,
                os.path.join(LOGS, f"preds.bayesian.test.{writer_name + writer_suff}.json"),
                os.path.join(LOGS, f"nbestpreds.bayesian.test.{writer_name + writer_suff}.json"),
                None, True, False, NULL_SCORE_THRESH, tokenizer,
            )

            results      = squad_evaluate(test_examples, predictions)
            report.em    = results["exact"]
            report.f1    = results["f1"]
            report.total = results["total"]
            
            print(f'em={report.em}, f1={report.f1}, total={report.total}')
            writer.add_scalar("bayesian_test_em",    report.em,    epoch)
            writer.add_scalar("bayesian_test_f1",    report.f1,    epoch)
            writer.add_scalar("bayesian_test_total", report.total, epoch)

    # ============================ SAVE =======================================

    torch.save({
        "weight_decay": WEIGHT_DECAY,
        "delta"       : DELTA,
        "model"       : b_model.state_dict(),
        "em"          : report.em,
        "f1"          : report.f1,
        "total"       : report.total,
    }, f"{writer_path + writer_suff}.pth")

    return report.acc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp",         type=str,   default="exp",                     help="experience name for logs")
    parser.add_argument("--model_name",  type=str,   default="distilbert-base-uncased", help="model name")
    parser.add_argument("--device",      type=str,   default="cuda:0",                  help="device (cpu, cuda:0, ..., cuda:n)")

    args = parser.parse_args()

    hypersearch = HyperSearch()
    hypersearch["DELTA"] = ((1e-2, 1e-1), True)
    hypersearch["WEIGHT_DECAY"] = ((1e-3, 0), False)
    
    score = hypersearch.search(
        train, iterations=10, EXP=args.exp, MODEL_NAME=args.model_name, DEVICE=args.device,
    )
    
    print("=========================== BEST SCORE ===========================")
    print(score)
    print("==================================================================")