import torch
from transformers import AutoTokenizer

def generate_ngrams(text,tokenizer, device, tokenwise=True, endtoend=True, split=1, n=100):
    ngrams = []
    
    if tokenwise:
        items = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)["input_ids"].squeeze(0).to(device)
    else:
        items = text.split()
    
    if endtoend:
        step = n
    else:
        step = split
    if n < 1:
        return []
    if len(items)<n:
        ngram=items
        if not tokenwise:
            ngram_text = ' '.join(ngram)
            ngram=tokenizer(f"Here is the context: {ngram_text}. Answer: ", return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].squeeze(0).to(device)
        else:
            ngram_text = tokenizer.decode(ngram)
            ngram=tokenizer(f"Here is the context: {ngram_text}. Answer: ", return_tensors="pt", padding=True,add_special_tokens=False)["input_ids"].squeeze(0).to(device)
        ngrams.append(ngram)
    else:
        for i in range(0, len(items) - n+1, step):
            if(i+step+n>len(items)):
                ngram = items[i:]
            else:
                ngram=items[i:i+n-1]
            
            if not tokenwise:
                ngram_text = ' '.join(ngram)
                ngram=tokenizer(f"Here is the context: {ngram_text}. Answer: ", return_tensors="pt", padding=True,add_special_tokens=False)["input_ids"].squeeze(0).to(device)
            else:
                ngram_text = tokenizer.decode(ngram)
                ngram=tokenizer(f"Here is the context: {ngram_text}. Answer: ", return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].squeeze(0).to(device)
            
            ngrams.append(ngram)
    ngram=tokenizer(f"Here is the context: (please pretend there is a context and try to summarize) Don't output something like: please provide some context. Answer: ", return_tensors="pt", padding=True, truncation=True)["input_ids"].squeeze(0).to(device)
    ngrams.append(ngram)
    return ngrams