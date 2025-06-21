import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import random
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image

random.seed(1337)


os.system('export CUDA_HOME=/usr/local/cuda-12.4')
os.system('export PATH=$CUDA_HOME/bin:$PATH')
os.system('export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH')

# model_id = "google/gemma-3-12b-it"
model_id = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

image_names = ['000.png', '010.png', '015.png', '020.png']
data_root = "media7link/gpt4point_test/"
random_ids = ['ba341c4ce89647ea9f6996ec58e3eacf', 'f93bb826f374423681a4772a3c49c1df', 'f8a87e1da0d54ed285c60cf5dc3d77e7', '58cd445c1e0044dd8af2009d51b7be18', '05068915ce654951910e905e24e35d38', '5bd2cda8deb04409bd0b4272966be972', '99565b63b54d429f99526428639037bf', '05068915ce654951910e905e24e35d38', '569b72c271c44106a3caa51f7d32dcd6']
body_html = ""

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

basic_object_description_path = "gpt4point_test_no_vec.json"
gpt4point_basic_descriptions = load_json(basic_object_description_path)

for i in range(len(random_ids)):
    images_paths = [os.path.join(data_root, random_ids[i], img_name) for img_name in image_names]
    images = [Image.open(p) for p in images_paths]

    gpt_basic_description = get_gpt_answer_by_object_id(gpt4point_basic_descriptions, random_ids[i])

    prompt = f"""You are a meticulous and precise visual analyst. Your task is to provide a single, factual, and objective paragraph describing the provided scene. Your description must be grounded exclusively in the visual information present in the images.

### Guiding Principles:
1. Describe, Don't Interpret: Report only what you see. Do not infer actions, intentions, history, or the contents of containers if they are not clearly visible. For example, if a box is closed, state that it is closed; do not guess its contents.
2. No Speculation: Avoid making assumptions. If you are uncertain about a material, describe its visual properties (e.g., "a dark, textured wood") rather than guessing a specific type (e.g., "oak"). If you cannot identify an object with certainty, describe its shape and color.
3. Literal and Unimaginative: Your goal is to be a camera, not a storyteller. Avoid creating a narrative or setting a mood. Stick to concrete, observable facts.

### Reference Description:
You are provided with the following basic description to use as a starting point. This description identifies the main subject(s).
"{gpt_basic_description}"

### Task:
Using the Reference Description to identify the main subjects, your task is to expand upon it. Based on the provided images and adhering strictly to the Guiding Principles above, generate a single, more detailed paragraph. Your paragraph should describe the main objects identified in the reference, their key attributes (color, shape, material, texture), and their spatial relationships to one another. The image is your sole source of truth. Focus on the primary subjects and their immediate surroundings, omitting details about the background."""

    question = """Generate the detailed, single-paragraph description."""

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt}]
        },
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": question}
            ]
        }
    ]

    # process with VLM
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    decoded = processor.decode(generation[0], skip_special_tokens=True)
    model_response = decoded.split('\nmodel\n', 1)[1]

    images_html = ""
    for imagepath in images_paths: images_html += f"""<img src="{imagepath}" style="width:200px;" alt="{imagepath}">\n"""
    output_html = f"""<pre class="pre-wrap">{model_response}</pre>"""
    body_html += images_html + f"""\n<pre class="pre-wrap">{gpt_basic_description}</pre>""" + f"\n<p>ID: {random_ids[i]}</p>\n" + output_html + "\n"

full_html_path = "gemma27_new_aug_desc"

gpt_basic_description = "basic_description"

full_html = f"""
<!DOCTYPE html>
<html>
<head>
<title>{full_html_path}</title>
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
<pre class="pre-wrap">{prompt}\n{question}</pre>
<h2>Outputs:</h2>
{body_html}
</body>
</html> 
"""

with open(f"{full_html_path}.html", "w", encoding="utf-8") as f:
    f.write(full_html)