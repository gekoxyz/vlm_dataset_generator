import json
import os
import random

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

random_id = random.choice(object_ids)
print(random_id)

from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
import torch

from PIL import Image

os.system('export CUDA_HOME=/usr/local/cuda-12.4')
os.system('export PATH=$CUDA_HOME/bin:$PATH')
os.system('export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH')

model_id = "google/gemma-3-12b-it"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

device = 'cuda:2'
model = Gemma3ForConditionalGeneration.from_pretrained(
    # model_id, low_cpu_mem_usage=True, quantization_config=bnb_config, device_map=device
    model_id, low_cpu_mem_usage=True, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

image_names = ['000.png', '010.png', '015.png', '020.png']
data_root = "media7link/gpt4point_test/"
random_id = "ba341c4ce89647ea9f6996ec58e3eacf"

body_html = ""

random_id = "ba341c4ce89647ea9f6996ec58e3eacf"

for i in range(10):
    images_paths = [os.path.join(data_root, random_id, img_name) for img_name in image_names]
    images = [Image.open(p) for p in images_paths]

    question = """
    Describe the image in detail. Focus on what can be clearly seen.

    Include:
    - Object appearance (color, shape, texture, material)
    - Spatial relationships between visible parts (e.g., “on the left”, “next to”)
    - Any visible components, patterns, or symbols

    Do not guess the function or identity unless it's visually obvious.
    Use neutral, objective language.

    Output:
    Description: <your detailed visual description here>
    """

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": question}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    decoded = processor.decode(generation[0], skip_special_tokens=True)
    model_response = decoded.split('\nmodel\n', 1)[1]

    images_html = ""

    for imagepath in images_paths:
        images_html += f"""<img src="{imagepath}" style="width:200px;" alt="{imagepath}">\n"""

    output_html = f"""<pre class="pre-wrap">{model_response}</pre>"""

    body_html += images_html + f"\n<p>ID: {random_id}</p>\n" + output_html + "\n"
    random_id = random.choice(object_ids)

full_html_path = "gemmagen_gpt_noquant.html"

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

with open(full_html_path, "w", encoding="utf-8") as f:
    f.write(full_html)