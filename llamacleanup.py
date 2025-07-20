import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

import torch
import json
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

os.system('export CUDA_HOME=/usr/local/cuda-12.4')
os.system('export PATH=$CUDA_HOME/bin:$PATH')
os.system('export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH')

model_id = "meta-llama/Llama-3.3-70B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
# Add a padding token if it doesn't exist. This is crucial for batching.
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Compiling model... (this may take a moment)")
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
print("Model compiled.")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

prompt_template = """You are a question-and-answer (QnA) analyzer.
Your task is to clean up a list of QnA pairs by applying the following rules:

### QnA constraints
1. Remove duplicate questions.
2. Remove questions related to the following topics:
   - Images with black backgrounds
   - Ambiguous references that depend on the viewer's point of view (e.g., “to the left of,” “to the right of”)
   - Obvious questions with trivial answers (e.g., Q: “Where is the left wheel of the car?” A: “On the left.”)

Note: Spatial terms that are viewpoint-independent (e.g., “next to,” “on top of,” “under”) are allowed.

### Output format
The output format must be the same as the input.
Don't provide any explaination about what you removed or why you removed it.

Keep only clear, meaningful, and non-redundant QnA pairs."""

DESCRIPTIONS_JSON_PATH = "llava72_desc_qna"
def load_json(file_path):
    with open(file_path, 'r') as file: return json.load(file)

descriptions = load_json(f"{DESCRIPTIONS_JSON_PATH}.json").get("items", [])

all_conversations = []
for item in descriptions:
    user_question = f"""Cleanup the following QnAs: {item.get("generated_qnas", "")}"""
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user",   "content": user_question},
    ]
    all_conversations.append(messages)

BATCH_SIZE = 1

print(f"Starting generation for {len(all_conversations)} items with batch size {BATCH_SIZE}...")

outputs = pipe(
    all_conversations,
    max_new_tokens=1024,
    batch_size=BATCH_SIZE,
    return_full_text=False,
    # Common generation parameters to ensure good output
    do_sample=False,
    # temperature=0.6,
    # top_p=0.9,
)

for i, item in enumerate(tqdm(descriptions, desc="Assigning results")): item["clean_qnas"] = outputs[i][0]['generated_text']

# --- Save Results ---
prompts = load_json(f"{DESCRIPTIONS_JSON_PATH}.json").get("prompt", [])
prompts.append(prompt_template)
generated_content_clean_qna = {"prompt" : prompts, "items": descriptions}

with open(f"{DESCRIPTIONS_JSON_PATH}_clean.json", 'w') as f: json.dump(generated_content_clean_qna, f, indent=2)

print("Processing complete. Output saved.")