import os
import random
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image

random.seed(1337)


# object_ids = [annotation['object_id'] for annotation in annotations_qa]
# object_ids = list(set(object_ids))
# random_id = random.choice(object_ids)

os.system('export CUDA_HOME=/usr/local/cuda-12.4')
os.system('export PATH=$CUDA_HOME/bin:$PATH')
os.system('export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH')

model_id = "google/gemma-3-12b-it"

device = 'cuda:2'
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, low_cpu_mem_usage=True, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id)

image_names = ['000.png', '010.png', '015.png', '020.png']
data_root = "media7link/gpt4point_test/"
random_ids = ['ba341c4ce89647ea9f6996ec58e3eacf', 'f93bb826f374423681a4772a3c49c1df', 'f8a87e1da0d54ed285c60cf5dc3d77e7', '58cd445c1e0044dd8af2009d51b7be18', '05068915ce654951910e905e24e35d38', '5bd2cda8deb04409bd0b4272966be972', '99565b63b54d429f99526428639037bf', '05068915ce654951910e905e24e35d38', '569b72c271c44106a3caa51f7d32dcd6']
body_html = ""

for i in range(len(random_ids)):
    images_paths = [os.path.join(data_root, random_ids[i], img_name) for img_name in image_names]
    images = [Image.open(p) for p in images_paths]

    prompt = """
    You are an expert image analyst. Your task is to provide a single, comprehensive, and detailed paragraph describing the scene presented in the following series of images.

    Focus on creating a rich description that would be useful for someone who cannot see the images. Pay close attention to:
    1. Main subject: clearly identify the primary object or character.
    2. Key attributes: describe its color, shape, size, texture, material, and any specific markings, text, or logos.
    3. Focusing on the foreground: describe just the main objects in the scene and the scene content, not its backgound.
    4. Spatial relationships: focus on the relative positions of the parts of the object.
    5. State or action: note if the subject is in a particular state (e.g., broken, shiny, old) or performing an action.

    Your entire output should be a single, well-written descriptive paragraph.
    """

    question = """
    Based on the provided images, please generate a detailed description for what you see.
    """

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
    ).to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        generation = model.generate(**inputs, do_sample=False)

    decoded = processor.decode(generation[0], skip_special_tokens=True)
    model_response = decoded.split('\nmodel\n', 1)[1]

    images_html = ""
    for imagepath in images_paths: images_html += f"""<img src="{imagepath}" style="width:200px;" alt="{imagepath}">\n"""
    output_html = f"""<pre class="pre-wrap">{model_response}</pre>"""
    body_html += images_html + f"\n<p>ID: {random_ids[i]}</p>\n" + output_html + "\n"

full_html_path = "gemma_noquant_notemp"

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
<pre class="pre-wrap">{prompt}\n\n{question}</pre>
<h2>Outputs:</h2>
{body_html}
</body>
</html> 
"""

with open(f"{full_html_path}.html", "w", encoding="utf-8") as f:
    f.write(full_html)