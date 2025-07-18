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

    prompt = f"""You are a meticulous and precise scene decomposition engine. Your task is to analyze the scene depicted in the provided images and output a structured breakdown of its contents. You must adhere strictly to what is visually observable. Do not infer, guess, or assume any details that are not explicitly visible.

You may optionally refer to the supplementary description below to disambiguate or more precisely name what is clearly visible. However, **all descriptions must remain grounded in visual evidence only**.

Supplementary description: "{basic_description}"

### Output Format

### Object Inventory  

List every distinct, clearly visible, primary object in the foreground of the scene. Use precise terminology (e.g., "armchair," "floor lamp," "coffee table"). Only include objects that are sufficiently visible to describe in detail. Don't make the decompositions too fine or coarse.
- [Object 1 Name]  
- [Object 2 Name]  
- [Object 3 Name]  
...

### Detailed Descriptions  
For each object listed above, describe the following attributes if they are visible:  
- Color  
- Shape  
- Material  
- Texture  
- State (e.g. new, worn, dusty, chipped)  
- Any visible markings, text, or logos  
- Number of components (e.g. sandwich with 4 bread slices, pizza with 12 pepperoni, office floor with 4 chairs)

If an attribute is not visually verifiable, omit it rather than speculate. Refer to the supplementary description only to resolve naming ambiguities when the visual evidence supports it.

- [Object 1 Name]: [Detailed visual attributes as above.]  
- [Object 2 Name]: [Detailed visual attributes as above.]  
...

### Spatial Relationships  
Describe the spatial layout and relative positions of the objects listed above using **the reference frame of each object**. This means describing other objects relative to how they would appear **from the point of view of the object itself**, rather than from the viewer/camera perspective.

Use clear and consistent prepositions (e.g., "to the left of," "on top of," "behind") relative to the object's orientation. If an object has an identifiable front, sides, or top/bottom based on its form, use that intrinsic orientation to describe spatial relationships.

Include both pairwise relationships and global placements within the scene. If an object appears in multiple positions (e.g., stacked, grouped), describe that clearly.

- [Object 1] has [Object 2] to its left (from Object 1's point of view).  
- [Object 3] rests on top of [Object 4], centered relative to its upper surface.  
- The [Object 5] is positioned along the back-left corner when considering its own front-facing side.  
...

Be unambiguous and specific. Prioritize spatial relationships that are important for understanding the physical layout of the scene and how objects are arranged relative to each other from their own orientations.
"""

    question = """Please analyze the scene using the given instructions."""

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

output_filename = "gemma27_decoCOUNTER"

basic_description = "BASIC_DESCRIPTION"
generated_content_wprompt = {
    "prompt": [f"{prompt}\n{question}"],
    "items": generated_content
}

with open(f"{output_filename}.json", 'w') as f:
    json.dump(generated_content_wprompt, f, indent=2)