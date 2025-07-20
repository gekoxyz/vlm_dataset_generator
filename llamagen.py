import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

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

prompt_template = """You are a meticulous and precise multiple choice QnA generator.
Your task is to provide three multiple choice questions and their corresponding answers based solely on a detailed scene description that will be provided to you.

### QnA constraints
1. Use only the information explicitly present in the scene description. Do not infer or assume anything beyond what is stated.
2. Each question must be based on spatial aspects of the scene such as positions, orientations, or relationships between objects (e.g. "next to" "above", "near", etc.) without using ambiguous terms such as "to the left/right of".
4. Each question must have exactly one correct answer.
5. All answer choices must be semantically distinct; avoid rewordings or rearrangements that convey the same meaning.
6. Ensure that no two questions focus on the same spatial detail, object relationship or topic. Like the color of a part of an object, position of a part of an object or other spatial features.
7. Topics to avoid: Don't ask question about these topics. They are excluded because they lead to bad questions. Avoid asking about: objects material. 

### Output Format
Format each of the three questions exactly as shown below:
Q: [Your Question Text Here]  
1. [Option 1 Text]  
2. [Option 2 Text]  
3. [Option 3 Text]  
A: [Correct Option Number]. [Full Text of Correct Option]

### Task
Based on the provided description and following the above guidelines and format, generate five (5) high-quality multiple choice QnAs."""

DESCRIPTIONS_JSON_PATH = "llava72_desc"
def load_json(file_path):
    with open(file_path, 'r') as file: return json.load(file)

descriptions = load_json(f"{DESCRIPTIONS_JSON_PATH}.json").get("items", [])

all_conversations = []
for item in descriptions:
    user_question = f"""Generate the QnA for the following description: {item.get("augmented_description", "")}"""
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user",   "content": user_question},
    ]
    all_conversations.append(messages)

BATCH_SIZE = 32

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

for i, item in enumerate(tqdm(descriptions, desc="Assigning results")):
    generated_qna = outputs[i][0]['generated_text']
    item["generated_qnas"] = generated_qna

# --- Save Results ---
prompts = load_json(f"{DESCRIPTIONS_JSON_PATH}.json").get("prompt", [])
prompts.append(prompt_template)
generated_content_wqna = {"prompt" : prompts, "items": descriptions}

with open(f"{DESCRIPTIONS_JSON_PATH}_qna.json", 'w') as f:
    json.dump(generated_content_wqna, f, indent=2)

print("Processing complete. Output saved.")