import subprocess
import csv
import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
import torchmetrics
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
import os
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer
from pmixedcopy import PMixED
from dpicl_get_results import get_results

np.random.seed(42)



epsilons = [2,4,8,16,32,64,128,256]
n_lengths = [64, 128, 256, 512]
query_budgets = [32, 48, 64]
alphas = [3, 4, 5, 6]
#p_s = [0.02, 0.03, 0.05, 0.1, 0.25, 0.5]
Ex_nums=[2.0, 3.0, 5.0, 10.0, 20.0]
default_epsilon = 8
default_query_budget = 48
default_n_length=256
default_Ex_num = 3.0
#default_iters = 2**3
default_alpha = 2
delta = 1e-5
endtoend=False
model_id = "google/gemma-2-2b-it"
access_token = "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"
cache_dir = "/home/woodygan/.cache"
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          padding_side="left",
                                          use_fast=True,
                                          token=access_token,
                                          cache_dir=cache_dir,
                                          trust_remote_code=True,)
    if tokenizer.pad_token is None:
        print("True")
        tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(model_id,trust_remote_code=True,torch_dtype=torch.float16,cache_dir=cache_dir)
    for epsilon in tqdm(epsilons, desc="Epsilon"):
        get_results(epsilon, default_query_budget, default_alpha, default_Ex_num, default_n_length, endtoend, delta, base_model, tokenizer,device)

