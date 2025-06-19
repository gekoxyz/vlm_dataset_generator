import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import random
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from transformers import pipeline
import torch

random.seed(1337)


os.system('export CUDA_HOME=/usr/local/cuda-12.4')
os.system('export PATH=$CUDA_HOME/bin:$PATH')
os.system('export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH')

model_id = "llava-hf/llava-interleave-qwen-7b-hf"

llava = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto"
).eval()

llava_processor = AutoProcessor.from_pretrained(model_id)

image_names = ['000.png', '010.png', '015.png', '020.png']
data_root = "media7link/gpt4point_test/"
random_ids = ['ba341c4ce89647ea9f6996ec58e3eacf','8361ad3d183843e885f58d1c68720771','214671c96c5f49b2a1927d1638f0fb47','3674ea1aabf9458dadd8332872509749','e348789bde904c2c87b99aae573637e4','dec1bb1c2b85451183f33066311e73a8','d0e07b22f1d54b968943e7a896235a65','797a7dfd60534ac4956428496f2cdae1','44795759d6144f61990796c02088665f','1e488ff902e34e62affd7961c88293bb','98c29f77095b45a9ad0a4e3014d111c6','e68820e2d14a46a08e23070e28c84b7b','7c00eea07b004402ac5b63ace4b2b78f','c4a0c2e2fb624bc0af9928b0ae6407ff','44795759d6144f61990796c02088665f','73d7b6f9a0b7410b945205338b090566','44795759d6144f61990796c02088665f','d611fdfc1ce945de86fb319587c35cf1','7c00eea07b004402ac5b63ace4b2b78f','b18a7d6210ca466f9dd9ceb8e1675a58']
body_html = ""

for i in range(len(random_ids)):
    images = [os.path.join(data_root, random_ids[i], img_name) for img_name in image_names]

    question = """You are a meticulous and precise visual analyst. Your task is to provide a single, factual, and objective paragraph describing the provided scene. Your description must be grounded exclusively in the visual information present in the images.

### Guiding Principles:
1. Describe, Don't Interpret: Report only what you see. Do not infer actions, intentions, history, or the contents of containers if they are not clearly visible. For example, if a box is closed, state that it is closed; do not guess its contents.
2. No Speculation: Avoid making assumptions. If you are uncertain about a material, describe its visual properties (e.g., "a dark, textured wood") rather than guessing a specific type (e.g., "oak"). If you cannot identify an object with certainty, describe its shape and color.
3. Literal and Unimaginative: Your goal is to be a camera, not a storyteller. Avoid creating a narrative or setting a mood. Stick to concrete, observable facts.

### Task:
Based on the provided images and adhering strictly to the principles above, generate a single paragraph. This paragraph should describe the main objects, their key attributes (color, shape, material, texture), and their spatial relationships to one another. Focus on the primary subjects and their immediate surroundings. Omit details about the distant or out-of-focus background.

Generate the descriptive paragraph based only on what you can see.
"""

    conversation = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": img} for img in images],
                            {"type": "text", "text": question}
                        ],
                    }
                ]

    # process with VLM
    inputs = llava_processor.apply_chat_template(
        [conversation], 
        add_generation_prompt=True, 
        tokenize=True, 
        return_dict=True, 
        return_tensors="pt"
    )
    
    # Generate with optimized parameters
    generate_ids = llava.generate(**inputs, max_new_tokens=512, do_sample=False) # do_sample=True, temperature=0.05)
    
    outputs = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    model_response = outputs[0].split('\nassistant\n', 1)[1]

    images_html = ""
    for imagepath in images: images_html += f"""<img src="{imagepath}" style="width:200px;" alt="{imagepath}">\n"""
    output_html = f"""<pre class="pre-wrap">{model_response}</pre>"""
    body_html += images_html + f"\n<p>ID: {random_ids[i]}</p>\n" + output_html + "\n"

full_html_path = "llava7"

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
<pre class="pre-wrap">{question}</pre>
<h2>Outputs:</h2>
{body_html}
</body>
</html> 
"""

with open(f"{full_html_path}.html", "w", encoding="utf-8") as f:
    f.write(full_html)
