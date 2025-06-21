import torch
import json
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# build the pipeline
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

prompt = """You are a meticulous and precise multiple choice QnA generator.
Your task is to provide three multiple choice questions and their corresponding answers based solely on a detailed scene description that will be provided to you.

### Guidelines
1. Use only the information explicitly present in the description to create each question and its correct answer. Don't ask question about the image background.
2. Each question must have exactly one correct, unambiguous answer.
3. All three answer options must be semantically distinct (e.g., avoid using "black and red" and "red and black" as separate options, since they are effectively the same).

### Output Format
Format each of the three questions exactly as shown below:
Q: [Your Question Text Here]  
1. [Option 1 Text]  
2. [Option 2 Text]  
3. [Option 3 Text]  
A: [Correct Option Number]. [Full Text of Correct Option]

### Task
Based on the provided description and following the above guidelines and format strictly, generate three high-quality multiple choice QnAs.
"""

DESCRIPTIONS_JSON_PATH = "llava7_desc.json"
def load_json(file_path):
    with open(file_path, 'r') as file: return json.load(file)

descriptions = load_json(DESCRIPTIONS_JSON_PATH).get("items", [])

body_html = ""

# for item in descriptions:
item = descriptions[0]
for i in range(2):
    question = f"""Generate the QnA for the following description: {item.get("augmented_description"), ""}"""

    # example conversation
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user",   "content": question},
    ]

    # run generation
    outputs = pipeline(
        messages,
        max_new_tokens=512,
    )

    item["generated_qnas"] = outputs[0]["generated_text"][2]["content"]

prompts = [load_json(DESCRIPTIONS_JSON_PATH).get("prompt", []), prompt]
generated_content_wqna = {"prompt" : prompts, "items": descriptions}

output_filename = "llava7_desc_qna"
with open(f"{output_filename}.json", 'w') as f:
    json.dump(generated_content_wqna, f, indent=2)