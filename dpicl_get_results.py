import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import torchmetrics
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
import os
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer
from pmixedcopy import PMixED

np.random.seed(42)

top_k = 50
top_p = 0.9
temp = 0.8
min_new_tokens = 15
do_sample = True
num_beams = 1
testsize=10
dataset_name = "xsum"
model_id = "google/gemma-2-2b-it"
max_input_length = 2048
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
access_token = "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"
cache_dir = "/home/woodygan/.cache"
class Evaluator:
    def __init__(self, metrics=None):
        if not metrics:
            metrics = ["rouge"]
        self.metrics = metrics
    
    def evaluate(self, predictions, references, documents, metrics=["rouge"]):
        result_dict = OrderedDict()
        if "rouge" in metrics:
            rouge_dict = self.calculate_rouge(predictions, references)
            for k, v in rouge_dict.items():
                result_dict[k] = v
        if "sacre_bleu" in metrics:
            sacre_bleu_dict = self.calculate_sacrebleu(predictions, references)
            for k, v in sacre_bleu_dict.items():
                result_dict[k] = v
        if "bertscore" in metrics:
            bertscore_dict = self.calculate_bertscore(predictions, references)
            for k, v in bertscore_dict.items():
                result_dict[k] = v
        if "factkb" in metrics:
            result_dict["factkb"] = self.calculate_factkb(predictions, documents)
        if "alignscore" in metrics:
            result_dict["alignscore"] = self.calculate_alignscore(predictions, documents)

        for k, v in result_dict.items():
            print(f"{k} -> {v*100:.2f}")
        return result_dict

    def calculate_rouge(self, predictions, references):
        from torchmetrics.functional.text.rouge import rouge_score
        rouge_dict = rouge_score(preds=predictions, target=references)
        return {k: v.item() for k, v in rouge_dict.items()}

    def calculate_sacrebleu(self, predictions, references):
        from torchmetrics.functional.text import sacre_bleu_score
        score = sacre_bleu_score(preds=predictions, target=[[i] for i in references])
        return {"sacre_bleu": score.item()}

    def calculate_bertscore(self, predictions, references):
        import evaluate
        bertscore = evaluate.load("bertscore")
        bertscore_dict = bertscore.compute(predictions=predictions, references=references, model_type="roberta-large-mnli")
        res = {"bertscore_precision": np.mean(bertscore_dict["precision"]), "bertscore_recall": np.mean(bertscore_dict["recall"]), "bertscore_f1": np.mean(bertscore_dict["f1"])}
        return {k: v.item() for k, v in res.items()}
    
    def calculate_alignscore(self, predictions, documents):
        from AlignScore.src.alignscore import AlignScore
        ckpt_path = "models/AlignScore-base.ckpt"
        align_scorer = AlignScore(model='roberta-base', batch_size=8, device=DEVICE, ckpt_path=ckpt_path, evaluation_mode='nli_sp')
        alignscore_result = align_scorer.score(contexts=documents, claims=predictions)
        return np.mean(alignscore_result)

    def calculate_factkb(self, predictions, documents):
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True, cache_dir=cache_dir)
        model = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", torch_dtype=torch.float16, cache_dir=cache_dir)
        model = model.to(DEVICE)
        res = []
        for i in range(len(predictions)):
            input_pretokenized = f"{predictions[i]} {tokenizer.sep_token} {documents[i]}"
            tokenized_input = tokenizer(input_pretokenized, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                output = model(input_ids=tokenized_input.input_ids.to(DEVICE))
            logits = torch.softmax(output.logits, dim=1)  # (bz, 2)
            res.append(logits.squeeze()[-1].item())
        return np.mean(res)

# Utility functions

def xsum_pretokenize(dataset, tokenizer, max_input_length):
    data = {"context": [], "query": [], "summary": []}
    for i, row in tqdm(enumerate(dataset), desc="truncating documents..."):
        trunc_doc = tokenizer.batch_decode(tokenizer(row['document'], return_tensors="pt", max_length=max_input_length,  truncation=True).input_ids, skip_special_tokens=True)[0]
        data['context'].append(trunc_doc)
        data['summary'].append(row['summary'])
        data["query"].append("You are a helpful assistant that summarizes text. Summarize the following article in one sentence.")
    return Dataset.from_dict(data)

def cnn_pretokenize(dataset, tokenizer, max_input_length):
    data = {"context": [], "query": [], "summary": []}
    for i, row in tqdm(enumerate(dataset), desc="truncating documents..."):
        trunc_doc = tokenizer.batch_decode(tokenizer(row['article'], return_tensors="pt", max_length=max_input_length,  truncation=True).input_ids, skip_special_tokens=True)[0]
        data['context'].append(trunc_doc)
        data['summary'].append(row['highlights'])
        data['query'].append("You are a helpful assistant that summarizes text. Summarize the following article in one sentence.")
    return Dataset.from_dict(data)

def pubmedqa_pretokenize(dataset, tokenizer, max_input_length):
    data = {"context": [], "query": [], "summary": []}
    for i, row in tqdm(enumerate(dataset), desc="truncating documents..."):
        context = ''.join(c for c in row['context']['contexts'])
        trunc_doc = tokenizer.batch_decode(tokenizer(context, return_tensors="pt", max_length=max_input_length, truncation=True).input_ids, skip_special_tokens=True)[0]
        data['context'].append(trunc_doc)
        data['summary'].append(row['long_answer'])
        data['query'].append(f"Question: {row['question']}. Answer:")
    return Dataset.from_dict(data)

def pretokenize(dataset_name, dataset, tokenizer, max_input_length):
    if dataset_name == "xsum":
        return xsum_pretokenize(dataset, tokenizer, max_input_length)
    elif dataset_name == "cnn":
        return cnn_pretokenize(dataset, tokenizer, max_input_length)
    elif dataset_name == "PubMedQA":
        return pubmedqa_pretokenize(dataset, tokenizer, max_input_length)
    return None

def template_input(row, dataset):
    if dataset == "xsum" or dataset == "cnn":
        return f"Article: {row['context']}. {row['query']}"
    elif dataset == "PubMedQA":
        return f"Document: {row['context']}. {row['query']}"
    else:
        return ""

def template_empty_input(row, dataset):
    if dataset == "xsum" or dataset == "cnn":
        return f"Article: . {row['query']}"
    elif dataset == "PubMedQA":
        return f"Document: . {row['query']}"
    else:
        return ""

def partition(data, tokenizer, partition_length, dataset_name):
    document_ids = tokenizer(data['context']).input_ids
    ensemble = []
    for i in range(0, len(document_ids), partition_length):
        idx = (i+partition_length)
        row = {'context': tokenizer.decode(document_ids[i:idx], skip_special_tokens=True), 'query': data['query']}
        ensemble.append(template_input(row, dataset_name))
    return ensemble

def partition_n_gram(data, tokenizer, dataset_name, n):
    document_ids = tokenizer(data['context']).input_ids
    length = len(document_ids)
    groups = []
    n_grams = []
    N = length - n + 1
    if N < 0:
        return [template_empty_input(data, dataset_name)]
    for i in range(N):
        removed_n_gram = document_ids[:i] + document_ids[i+n:]
        n_grams.append(document_ids[i:i+n])
        row = {'context': tokenizer.decode(removed_n_gram, skip_special_tokens=True), 'query': data['query']}
        groups.append(template_input(row, dataset_name))
    return groups, n_grams

def predict(test_set, ensemble, tokenizer, temperature, dataset_name, min_length, max_length, top_k, tokenwise=True, endtoend=False, split=1,n_length=128):
    predictions = []
    for idx, row in tqdm(enumerate(test_set), total=len(test_set)):       
        output=ensemble.priv_generate(row['context'], n_length, row['query'], temperature=temperature, min_length=min_length, max_length=max_length, top_k=None,tokenwise=tokenwise, endtoend=endtoend,split=split)
        output=tokenizer.decode(output, skip_special_tokens=True)
        print(output)
        predictions.append(output.split("Answer:")[-1].strip())
    
    return predictions


def get_results(epsilon, query_budget, alpha, Ex_num, n_length, endtoend,delta,model,tokenizer,device):
    base_model=model
    tokenizer=tokenizer
    ensemble_model = PMixED(base_model, model_id, tokenizer, 
                            device=device, q_budget=query_budget, alpha=alpha, delta=delta,Ex_num=Ex_num,
                            eps=epsilon, beta=0., lambd=None, 
                            threshold=None, screen_top_k=0, sigma=None, 
                            accounting_method="Independent")
    if dataset_name == "PubMedQA":
        raw_test_set = load_dataset("qiaojin/PubMedQA", "pqa_labeled", cache_dir=cache_dir)['train']
    elif dataset_name == 'xsum':
        raw_test_set = load_dataset(dataset_name, split=f"test[:{testsize}]", trust_remote_code=True)
    elif dataset_name == 'cnn':
        raw_test_set = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test[:20]", cache_dir=cache_dir)

    test_set = pretokenize(dataset_name, raw_test_set, tokenizer, max_input_length)

    dir_name = "results"
    os.makedirs(dir_name, exist_ok=True)

    predictions = predict(test_set, ensemble_model, tokenizer, temperature=temp, dataset_name=dataset_name, min_length=min_new_tokens,max_length=query_budget, top_k=top_k,endtoend=endtoend, n_length=n_length)
    documents, references = [], []
    for idx, data in tqdm(enumerate(test_set), total=len(test_set)):
        documents.append(data['context'])
        references.append(data['summary'])
    df = pd.DataFrame({'generations': predictions, 'references': references})
    file_name = f'pmixed-{dataset_name}-{testsize}-{model_id.split("/")[-1]}/Q_budget={query_budget}/alpha={alpha}/EtE={endtoend}/Ex_num={Ex_num}/n_length={n_length}/eps={epsilon}.csv'
    full_path = os.path.join(dir_name, file_name)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    df.to_csv(full_path)
    evaluator = Evaluator()
    result_dict = evaluator.evaluate(predictions, references, documents)
    result_dict.update({
        'Q_budget': query_budget,
        'alpha': alpha,
        'EtE': endtoend,
        'Ex_num': Ex_num,
        'n_length': n_length,
        'eps': epsilon
        })

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(result_dict, orient='index').transpose()

    # File name for evaluation results
    eval_file_name = f'evaluation-pmixed-{dataset_name}-{testsize}-{model_id.split("/")[-1]}-aggregate.csv'
    full_path = os.path.join(dir_name, eval_file_name)

    # Check if file exists and append or create new file
    if os.path.exists(full_path):
        # If file exists, read it first
        existing_df = pd.read_csv(full_path)
        # Append new results
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        # Save updated results
        updated_df.to_csv(full_path, index=False)
        print(f"Results appended to existing file: {eval_file_name}")
    else:
        df.to_csv(full_path, index=False)
        print(f"New results file created: {eval_file_name}")