import os
import json
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

os.system('export CUDA_HOME=/usr/local/cuda-12.4')
os.system('export PATH=$CUDA_HOME/bin:$PATH')
os.system('export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH')

torch.set_float32_matmul_precision('high')

model_id = "google/gemma-3-27b-it"

model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

image_names = ['000.png', '010.png', '015.png', '020.png']
data_root = "media7link/gpt4point_test/"

annotations_qa_root = '/media/data7/DATASET/shapenerf_objanerf_text/spatial_gpt4point_qa/texts'
with open(os.path.join(annotations_qa_root, 'spatial_gpt4point_qa_no_vec.json'), 'r') as f:
    annotations_qa = json.load(f)
object_ids = [annotation['object_id'] for annotation in annotations_qa]
object_ids = list(set(object_ids))

def load_json(file_path):
    with open(file_path, 'r') as file: return json.load(file)

def get_basic_description_by_object_id(data, object_id):
    for item in data:
        if item["object_id"] == object_id:
            for conversation in item["conversations"]:
                if conversation["from"] == "gpt":
                    return conversation["value"]
    return "[BASIC DESCRIPTION NOT AVAILABLE, FOCUS ONLY ON THE IMAGES]"

basic_object_description_path = "gpt4point_test_no_vec.json"
gpt4point_basic_descriptions = load_json(basic_object_description_path)

generated_content = []

for object_id in object_ids:
    images_paths = [os.path.join(data_root, object_id, img_name) for img_name in image_names]
    images = [Image.open(p) for p in images_paths]

    basic_description = get_basic_description_by_object_id(gpt4point_basic_descriptions, object_id)

    prompt = f"""You are a meticulous and precise scene decomposition engine. Your task is to analyze the provided images and output a structured description. Do not infer, guess, or assume any information not explicitly visible in the image.

You may optionally refer to a brief supplementary description that contains human-written notes about the image. However, your output must remain grounded in visual evidence only. Use the description solely to help you disambiguate or more precisely describe what is clearly visible.

Supplementary description: "{basic_description}"

### Output Format

### Object Inventory
List every distinct primary object in the foreground of the scene. Use precise terminology where possible (e.g., "armchair," "floor lamp," "coffee table").
- [Object 1 Name]
- [Object 2 Name]
- [Object 3 Name]
...

### Detailed Descriptions
For each object listed above, provide a detailed description of its attributes.
- **[Object 1 Name]:** [Describe color, shape, material, texture, state (e.g., new, dusty, chipped), and any visible text or logos. You may cross-reference the supplementary description only if the details are visually verifiable.]
- **[Object 2 Name]:** [Describe its attributes.]
- **[Object 3 Name]:** [Describe its attributes.]
...

### Spatial Relationships
Describe the positions of the objects relative to each other and to the overall scene. Use clear and simple prepositions.
- [Object 1] is located [e.g., to the left of Object 2].
- [Object 3] is positioned [e.g., on top of the coffee table].
- The [e.g., stack of magazines] is placed [e.g., next to the armchair on the floor].
- All objects are resting on a surface that appears to be [describe the floor or ground surface]."""

    question = """Please analyze the images using the given instructions."""

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
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    decoded = processor.decode(generation[0], skip_special_tokens=True)
    model_response = decoded.split('\nmodel\n', 1)[1]

    item_data = {
        "item_id": object_id,
        "basic_description": basic_description,
        "augmented_description": model_response
    }
    generated_content.append(item_data)

output_filename = "gemma27_decomposition"

basic_description = "BASIC_DESCRIPTION"
generated_content_wprompt = {
    "prompt": [f"{prompt}\n{question}"],
    "items": generated_content
}

with open(f"{output_filename}.json", 'w') as f:
    json.dump(generated_content_wprompt, f, indent=2)