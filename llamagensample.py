import torch
import json
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm

os.system('export CUDA_HOME=/usr/local/cuda-12.4')
os.system('export PATH=$CUDA_HOME/bin:$PATH')
os.system('export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"

# --- Model and Tokenizer Setup (with Optimizations) ---

model_id = "meta-llama/Llama-3.3-70B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
# Add a padding token if it doesn't exist. This is crucial for batching.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load model with 8-bit quantization and optimizations
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

# OPTIMIZATION: Use torch.compile for a JIT compilation speed boost (requires PyTorch 2.0+)
# The first run will be slow as it compiles, but subsequent runs will be much faster.
print("Compiling model... (this may take a moment)")
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
print("Model compiled.")

# build the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

prompt_template = """You are a meticulous and precise multiple choice QnA generator.
Your task is to provide three multiple choice questions and their corresponding answers based solely on the provided scene decomposition.

### Guidelines
1. Only base your questions on details explicitly stated in the scene decomposition. Do not infer or imagine any additional context, visual characteristics, or common-sense expectations not mentioned in the description.
2. Questions must concern *object-to-object* spatial facts (relative position, orientation, contact, containment, stacking, alignment, etc.).
3. Do not create questions about object colors, materials, or the scene background or base. Focus strictly on spatial features and object arrangements.
4. Each question must have exactly one correct answer. The answer must be non-obvious yet directly verifiable from the text (non-obvious e.g. a wheel attached to a car would be obvious, but a detached one would not be).
5. All answer choices must be semantically distinct. Avoid rewordings or rearrangements that convey the same meaning.
6. Ensure that no two questions focus on the same spatial detail or object relationship.
7. If any spatial detail is ambiguous or contradictory, omit it rather than speculating.

### Definitions
- "Primary objects" refer to the major components described in the scene that can be independently segmented or referenced. These are distinct from minor elements or attached parts unless those parts are mentioned with their own spatial positioning.

### Output Format
Format each of the three questions exactly as shown below:
Q: [Your Question Text Here]  
1. [Option 1 Text]  
2. [Option 2 Text]  
3. [Option 3 Text]  
A: [Correct Option Number]. [Full Text of Correct Option]

Do not add explanations or extra commentary outside the specified format. Follow the exact indentation and line spacing.

### Question Topics
- The first two questions must focus on spatial relationships between primary objects in the scene.
- The third question must involve counting the number of objects or spatially distinct parts of an object (if such numerical information is available).
- If the scene decomposition lacks any countable elements or numeric detail, only then fallback to a third spatial question, still adhering strictly to the above guidelines.

### Task
Based on the provided scene decomposition and following the above guidelines and format, generate three high-quality multiple choice QnAs.
"""

DESCRIPTIONS_JSON_PATH = "gemma27_decov5sample"
def load_json(file_path):
    with open(file_path, 'r') as file: return json.load(file)

descriptions = load_json(f"{DESCRIPTIONS_JSON_PATH}.json").get("items", [])

# --- BATCH PREPARATION ---
# 1. Create a list of all conversations to be processed
all_conversations = []
for item in descriptions:
    user_question = f"""Generate the QnA for the following description: {item.get("augmented_description", "")}"""
    messages = [
        {"role": "system", "content": prompt_template},
        {"role": "user",   "content": user_question},
    ]
    all_conversations.append(messages)

# --- BATCH INFERENCE ---
# 2. Run the pipeline ONCE on the entire list.
# Adjust batch_size based on your VRAM. Start with 8 or 16 and increase
# until you get a CUDA out-of-memory error, then reduce slightly.
BATCH_SIZE = 2 # TUNE THIS PARAMETER

print(f"Starting generation for {len(all_conversations)} items with batch size {BATCH_SIZE}...")

# The pipeline automatically handles the chat template for you.
# By setting return_full_text=False, we only get the assistant's generated reply.
outputs = pipe(
    all_conversations,
    max_new_tokens=1024,
    batch_size=BATCH_SIZE,
    return_full_text=False, # Simplifies output parsing
    # Common generation parameters to ensure good output
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

# --- POST-PROCESSING ---
# 3. Assign the generated QnAs back to your original data structure
for i, item in enumerate(tqdm(descriptions, desc="Assigning results")):
    # The output is now a list of generated texts.
    # Each element corresponds to an input conversation.
    # The structure is simpler because of return_full_text=False
    generated_qna = outputs[i][0]['generated_text']
    item["generated_qnas"] = generated_qna

# --- Save Results ---
prompts = [load_json(f"{DESCRIPTIONS_JSON_PATH}.json").get("prompt", []), prompt_template]
generated_content_wqna = {"prompt" : prompts, "items": descriptions}

with open(f"{DESCRIPTIONS_JSON_PATH}_qna.json", 'w') as f:
    json.dump(generated_content_wqna, f, indent=2)

print("Processing complete. Output saved.")