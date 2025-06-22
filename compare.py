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
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

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


GEMMA_JSON_PATH = "gemma27_desc.json"
LLAVA_JSON_PATH = "llava7_desc.json"
def load_json(file_path):
    with open(file_path, 'r') as file: return json.load(file)

gemma27_descriptions = load_json(GEMMA_JSON_PATH).get("items", [])
llava7_descriptions = load_json(LLAVA_JSON_PATH).get("items", [])

gemma27_descriptions.sort(key=lambda item: item['item_id'])
llava7_descriptions.sort(key=lambda item: item['item_id'])


differences = []

for i in range(len(gemma27_descriptions)):
    gemma_item_id = gemma27_descriptions[i]["item_id"]
    llava_item_id = llava7_descriptions[i]["item_id"]

    print(gemma_item_id == llava_item_id)

    gemma27_detailed_description = gemma27_descriptions[i]["augmented_description"]
    llava7_detailed_description = llava7_descriptions[i]["augmented_description"]

    prompt = f"""# ROLE
You are an expert text comparison agent. Your purpose is to meticulously analyze two descriptions of the same scene or object and identify all semantic differences between them.

# TASK
You will be given two text descriptions, labeled [DESCRIPTION 1] and [DESCRIPTION 2]. Your task is to return both descriptions with the specific words or short phrases that constitute a semantic difference wrapped in bold HTML tags (`<b>...</b>`).

# RULES
1.  **Identify Semantic Differences:** Focus on differences in attributes (color, size, shape), quantity, state, actions, or the presence/absence of objects or details.
2.  **Highlight in BOTH Descriptions:** The corresponding differing elements must be highlighted in *both* of the provided texts.
3.  **Minimal Highlighting:** Be precise. Only bold the exact word(s) that create the difference. For example, if the difference is "a large red car" vs. "a small red car", you must highlight only "<b>large</b>" and "<b>small</b>", not the entire phrase.
4.  **Handle Presence/Absence:** If one description includes a detail that the other completely omits, bold the entire phrase describing that unique element in the text where it appears. There will be no corresponding tag in the other text for this specific difference.
5.  **Maintain Original Text:** Do not add, remove, or change any words from the original descriptions. Only insert the `<b>` and `</b>` tags.
6.  **Output Format:** Present the final output clearly, with each modified description on a new line.

# EXAMPLES

### Example 1: Simple Attribute Difference
[DESCRIPTION 1]: There is a red tractor in the field next to a barn.
[DESCRIPTION 2]: There is a blue tractor in the field next to a barn.

**Result:**
Description 1 with Highlights: There is a <b>red</b> tractor in the field next to a barn.
Description 2 with Highlights: There is a <b>blue</b> tractor in the field next to a barn.
---
### Example 2: Quantity Difference
[DESCRIPTION 1]: A photograph of a dog running on the beach.
[DESCRIPTION 2]: A photograph of two dogs running on the beach.

**Result:**
Description 1 with Highlights: A photograph of <b>a</b> dog running on the beach.
Description 2 with Highlights: A photograph of <b>two</b> dogs running on the beach.
---
### Example 3: Action/State Difference
[DESCRIPTION 1]: The woman is standing by the window, looking out at the rain.
[DESCRIPTION 2]: The woman is sitting by the window, reading a book.

**Result:**
Description 1 with Highlights: The woman is <b>standing</b> by the window, <b>looking out at the rain</b>.
Description 2 with Highlights: The woman is <b>sitting</b> by the window, <b>reading a book</b>.
---
### Example 4: Presence/Absence Difference
[DESCRIPTION 1]: A cozy living room with a fireplace, a large sofa, and a coffee table.
[DESCRIPTION 2]: A cozy living room with a fireplace, a large sofa, a coffee table, and a sleeping cat on the rug.

**Result:**
Description 1 with Highlights: A cozy living room with a fireplace, a large sofa, and a coffee table.
Description 2 with Highlights: A cozy living room with a fireplace, a large sofa, a coffee table, and <b>a sleeping cat on the rug</b>.
---
### Example 5: Multiple and Complex Differences
[DESCRIPTION 1]: A detailed oil painting depicts a serene coastal village at dawn. Three small fishing boats are moored in the calm harbor, and a lighthouse stands on the distant cliff.
[DESCRIPTION 2]: A detailed watercolor painting depicts a bustling coastal village at sunset. Several large fishing boats are sailing out of the turbulent harbor.

**Result:**
Description 1 with Highlights: A detailed <b>oil</b> painting depicts a <b>serene</b> coastal village at <b>dawn</b>. <b>Three small</b> fishing boats are <b>moored in</b> the <b>calm</b> harbor, and <b>a lighthouse stands on the distant cliff</b>.
Description 2 with Highlights: A detailed <b>watercolor</b> painting depicts a <b>bustling</b> coastal village at <b>sunset</b>. <b>Several large</b> fishing boats are <b>sailing out of</b> the <b>turbulent</b> harbor.

# YOUR TASK

Now, perform this task for the following two descriptions.

[DESCRIPTION 1]:
{gemma27_detailed_description}

[DESCRIPTION 2]:
{llava7_detailed_description}
"""

    # example conversation
    messages = [
        # {"role": "system", "content": prompt},
        {"role": "user",   "content": prompt},
    ]

    # run generation
    outputs = pipeline(
        messages,
        max_new_tokens=512,
    )

    out = outputs[0]["generated_text"][1]["content"]

    item = {
        "item_id": gemma_item_id,
        "differences": out
    }
    differences.append(item)


output_filename = "desc_differences"
with open(f"{output_filename}.json", 'w') as f:
    json.dump(differences, f, indent=2)