import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

output=pipeline("Hey how are you doing today? Answer:")
print(output[0]['generated_text'].split("Answer:")[-1])