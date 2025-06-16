import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt

random.seed(1337)

annotations_qa_root = '/media/data7/DATASET/shapenerf_objanerf_text/spatial_gpt4point_qa/texts'
with open(os.path.join(annotations_qa_root, 'spatial_gpt4point_qa_no_vec.json'), 'r') as f:
    annotations_qa = json.load(f)
print(len(annotations_qa))
print(annotations_qa[0])  

annotations_cap_root = '/media/data7/DATASET/shapenerf_objanerf_text/spatial_gpt4point_cap/texts'
with open(os.path.join(annotations_cap_root, 'spatial_gpt4point_cap.json'), 'r') as f:
    annotations_cap = json.load(f)
print(len(annotations_cap))
print(annotations_cap[0])

data_root = '/media/data7/DATASET/objanerf_text_evaluation_data/imgs_from_objaverse/gpt4point_test'

object_ids = [annotation['object_id'] for annotation in annotations_qa]
object_ids = list(set(object_ids))
print(len(object_ids))

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from transformers import pipeline
import torch

os.system('export CUDA_HOME=/usr/local/cuda-12.4')
os.system('export PATH=$CUDA_HOME/bin:$PATH')
os.system('export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH')

model_id = "llava-hf/llava-interleave-qwen-7b-hf"

device = 'cuda:2'
llava = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    low_cpu_mem_usage=True, 
    device_map="auto"
).eval()

llava_processor = AutoProcessor.from_pretrained(model_id)

random_ids = ["ba341c4ce89647ea9f6996ec58e3eacf"]
for i in range(12): random_ids.append(random.choice(object_ids))

image_names = ['000.png', '010.png', '015.png', '020.png']
data_root = "media7link/gpt4point_test/"
basic_object_description_path = "gpt4point_test_no_vec.json"

# ======================================== MANUAL DESC ===================================================================

import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_gpt_answer_by_object_id(data, object_id):
    for item in data:
        if item["object_id"] == object_id:
            for conversation in item["conversations"]:
                if conversation["from"] == "gpt":
                    return conversation["value"]
    return "Object ID not found."

gpt4point_basic_descriptions = load_json(basic_object_description_path)

body_html = ""

for i in range(10):
    gpt_basic_description = get_gpt_answer_by_object_id(gpt4point_basic_descriptions, random_ids[i])
    question = f"""
    Provide a very detailed description of the object in the images knowing that its caption is: "{gpt_basic_description}". Focus on the relative positions of the parts of the object. Include as many details as possible. Focus only on the content of the images. Don't focus on the background.
    """
    images = [os.path.join(data_root, random_ids[i], img_name) for img_name in image_names]

    # build conversation
    conversation = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": img} for img in images],
                            {"type": "text", "text": question}
                        ],
                    }
                ]

    # Process with VLM
    inputs = llava_processor.apply_chat_template(
        [conversation], 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(device)

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with optimized parameters
    # generate_ids = llava.generate(**inputs, max_new_tokens=128)
    generate_ids = llava.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.05)
    outputs = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    model_response = outputs[0].split('\nassistant\n', 1)[1]
    # model_response = outputs[0]

    images_html = ""

    for imagepath in images:
        images_html += f"""<img src="{imagepath}" style="width:200px;" alt="{imagepath}">\n"""

    output_html = f"""<pre class="pre-wrap">{model_response}</pre>"""

    body_html += images_html + f"\n<p>ID: {random_ids[i]}</p>\n" + output_html + "\n"

full_html_path = "llavagen_noquant.html"

full_html = f"""
<!DOCTYPE html>
<html>
<head>
<title>LLaVA generation output</title>
<style>
    .pre-wrap {{
    white-space: pre-wrap;   /* Preserves whitespace and wraps text */
    word-wrap: break-word;   /* Ensures long words break */
    max-width: 800px;          /* Maximum width (could also use px or rem) */
    }}
</style>
</head>
<body>
<h2>Prompt:</h2>
<pre class="pre-wrap">{question}</pre>
<h2>Outputs:</h2>
{body_html}
<h3>Annotations on the generation:</h3>

</body>
</html> 
"""

with open("llavagen_manual_desc.html", "w", encoding="utf-8") as f:
    f.write(full_html)

# ======================================== GEMINI DESC ===================================================================
body_html = ""

for i in range(10):
    gpt_basic_description = get_gpt_answer_by_object_id(gpt4point_basic_descriptions, random_ids[i])
    question = f"""
    You are an AI assistant specializing in description enhancement. You will be given a set of images and a brief, human-provided description.

    Your task is to use the brief description as a starting point and EXPAND it into a much more detailed and comprehensive paragraph by meticulously analyzing the visual information in the images.

    Your goal is to add new, valuable details that are not present in the original brief description, without making any mistake. Focus on:

    *   **Elaborating on the main subject:** Add specifics about its color, texture, material, and condition.
    *   **Focusing on the foreground**: Describe just the main objects in the scene and the scene content, not its backgound.
    *   **Specifying spatial relationships:** Clearly state where things are in relation to each other.
    *   **Adding interesting observations:** Note any unique features, text, logos, or unusual aspects visible in the images.

    **Example Input Format:**
    BRIEF DESCRIPTION: A car with a hat on it.
    [Your enhanced, detailed paragraph would go here, describing the type of car, its color, the type of hat, the street, etc.]

    **IMPORTANT:** Do NOT just repeat the brief description. Your primary job is to ADD information from the images. Do not write a question or a list. Your output must be a single, enhanced paragraph.

    ---
    **YOUR TASK**

    BRIEF DESCRIPTION: {gpt_basic_description}

    Now, using this brief description and the images provided, generate your enhanced, detailed description.
    """
    images = [os.path.join(data_root, random_ids[i], img_name) for img_name in image_names]

    # build conversation
    conversation = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": img} for img in images],
                            {"type": "text", "text": question}
                        ],
                    }
                ]

    # Process with VLM
    inputs = llava_processor.apply_chat_template(
        [conversation], 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(device)

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with optimized parameters
    # generate_ids = llava.generate(**inputs, max_new_tokens=128)
    generate_ids = llava.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.05)
    outputs = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    model_response = outputs[0].split('\nassistant\n', 1)[1]

    images_html = ""

    for imagepath in images:
        images_html += f"""<img src="{imagepath}" style="width:200px;" alt="{imagepath}">\n"""

    output_html = f"""<pre class="pre-wrap">{model_response}</pre>"""

    body_html += images_html + f"\n<p>ID: {random_ids[i]}</p>\n" + output_html + "\n"

full_html_path = "llavagen_noquant.html"

full_html = f"""
<!DOCTYPE html>
<html>
<head>
<title>LLaVA generation output</title>
<style>
    .pre-wrap {{
    white-space: pre-wrap;   /* Preserves whitespace and wraps text */
    word-wrap: break-word;   /* Ensures long words break */
    max-width: 800px;          /* Maximum width (could also use px or rem) */
    }}
</style>
</head>
<body>
<h2>Prompt:</h2>
<pre class="pre-wrap">{question}</pre>
<h2>Outputs:</h2>
{body_html}
<h3>Annotations on the generation:</h3>

</body>
</html> 
"""

with open("llavagen_gemini_desc.html", "w", encoding="utf-8") as f:
    f.write(full_html)



# ======================================== GPT DESC ===================================================================
body_html = ""

for i in range(10):
    gpt_basic_description = get_gpt_answer_by_object_id(gpt4point_basic_descriptions, random_ids[i])
    question = f"""
    You are given some images and a short human annotation.

    Annotation: "{gpt_basic_description}"

    Use both the image and the annotation to produce a more detailed visual description. Expand and clarify based on what is visible in the image. Do not copy or repeat the annotation directly.

    Focus on:
    - Visual features (shape, color, texture, material)
    - Relative positions and structure
    - Any visible text, symbols, or labels

    Now, using this brief annotation and the images provided, generate your enhanced, detailed description
    """
    images = [os.path.join(data_root, random_ids[i], img_name) for img_name in image_names]

    # build conversation
    conversation = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": img} for img in images],
                            {"type": "text", "text": question}
                        ],
                    }
                ]

    # Process with VLM
    inputs = llava_processor.apply_chat_template(
        [conversation], 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt"
    ).to(device)

    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with optimized parameters
    # generate_ids = llava.generate(**inputs, max_new_tokens=128)
    generate_ids = llava.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.05)
    outputs = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    model_response = outputs[0].split('\nassistant\n', 1)[1]

    images_html = ""

    for imagepath in images:
        images_html += f"""<img src="{imagepath}" style="width:200px;" alt="{imagepath}">\n"""

    output_html = f"""<pre class="pre-wrap">{model_response}</pre>"""

    body_html += images_html + f"\n<p>ID: {random_ids[i]}</p>\n" + output_html + "\n"

full_html_path = "llavagen_noquant.html"

full_html = f"""
<!DOCTYPE html>
<html>
<head>
<title>LLaVA generation output</title>
<style>
    .pre-wrap {{
    white-space: pre-wrap;   /* Preserves whitespace and wraps text */
    word-wrap: break-word;   /* Ensures long words break */
    max-width: 800px;          /* Maximum width (could also use px or rem) */
    }}
</style>
</head>
<body>
<h2>Prompt:</h2>
<pre class="pre-wrap">{question}</pre>
<h2>Outputs:</h2>
{body_html}
<h3>Annotations on the generation:</h3>

</body>
</html> 
"""

with open("llavagen_gpt_desc.html", "w", encoding="utf-8") as f:
    f.write(full_html)
