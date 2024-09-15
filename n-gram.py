import torch
from transformers import AutoTokenizer

def generate_ngrams(text, tokenizer, model_size, tokenwise=True, endtoend=True,length=50):
    tokenizer = tokenizer
    ngrams = []
    
    # Tokenize the entire text once
    full_tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Encode the constant parts
    context_prefix = tokenizer.encode("context:", add_special_tokens=False)
    question_suffix = tokenizer.encode(". question:", add_special_tokens=False)
    
    if tokenwise:
        items = full_tokens
    else:
        items = text.split()
    
    if endtoend:
        n = len(items) // model_size
        step = n
    else:
        n=length
        step = (len(items)-length) // (model_size-1)
    
    if n < 1:
        return []
    
    for i in range(0, len(items) - n + 1, step):
        if tokenwise:
            ngram = items[i:i+n]
        else:
            ngram_text = ' '.join(items[i:i+n])
            ngram = tokenizer.encode(ngram_text, add_special_tokens=False)
        
        formatted_ngram = context_prefix + ngram + question_suffix
        ngrams.append(formatted_ngram)
    
    return ngrams