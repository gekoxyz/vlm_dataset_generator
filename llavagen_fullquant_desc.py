import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import json
import random
from transformers import AutoProcessor, LlavaForConditionalGeneration
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

annotations_qa_root = '/media/data7/DATASET/shapenerf_objanerf_text/spatial_gpt4point_qa/texts'
with open(os.path.join(annotations_qa_root, 'spatial_gpt4point_qa_no_vec.json'), 'r') as f:
    annotations_qa = json.load(f)
object_ids = [annotation['object_id'] for annotation in annotations_qa]
object_ids = list(set(object_ids))

def load_json(file_path):
    with open(file_path, 'r') as file: return json.load(file)

def get_answer_by_object_id(data, object_id):
    for item in data:
        if item["object_id"] == object_id:
            for conversation in item["conversations"]:
                if conversation["from"] == "gpt": return conversation["value"]
    return "[BASIC DESCRIPTION NOT AVAILABLE, FOCUS ONLY ON THE IMAGES]"

basic_object_description_path = "gpt4point_test_no_vec.json"
gpt4point_basic_descriptions = load_json(basic_object_description_path)

generated_content = []

# for i in range(len(random_ids)):
for object_id in object_ids:
    # images = [os.path.join(data_root, random_ids[i], img_name) for img_name in image_names]
    images = [os.path.join(data_root, object_id, img_name) for img_name in image_names]
    
    basic_description = get_answer_by_object_id(gpt4point_basic_descriptions, object_id)

    question = f"""You are a meticulous and precise visual analyst. Your task is to provide a single, factual, and objective paragraph describing the provided scene. Your description must be grounded exclusively in the visual information present in the images.

### Guiding Principles:
1. Describe, Don't Interpret: Report only what you see. Do not infer actions, intentions, history, or the contents of containers if they are not clearly visible. For example, if a box is closed, state that it is closed; do not guess its contents.
2. No Speculation: Avoid making assumptions. If you are uncertain about a material, describe its visual properties (e.g., "a dark, textured wood") rather than guessing a specific type (e.g., "oak"). If you cannot identify an object with certainty, describe its shape and color.
3. Literal and Unimaginative: Your goal is to be a camera, not a storyteller. Avoid creating a narrative or setting a mood. Stick to concrete, observable facts.

### Reference Description:
You are provided with the following basic description to use as a starting point. This description identifies the main subject(s).
"{basic_description}"

### Task:
Using the Reference Description to identify the main subjects, your task is to expand upon it. Based on the provided images and adhering strictly to the Guiding Principles above, generate a single, more detailed paragraph. Your paragraph should describe the main objects identified in the reference, their key attributes (color, shape, material, texture), and their spatial relationships to one another. The image is your sole source of truth. Focus on the primary subjects and their immediate surroundings, omitting details about the background.

Generate your detailed, single-paragraph description.
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
    generate_ids = llava.generate(**inputs, max_new_tokens=512, do_sample=False)
    
    outputs = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    model_response = outputs[0].split('\nassistant\n', 1)[1]

    item_data = {
        # "item_id": random_ids[i],
        "item_id": object_id,
        "basic_description": basic_description,
        "augmented_description": model_response
    }

    generated_content.append(item_data)

output_filename = "llava7_desc"

basic_description = "BASIC_DESCRIPTION"
generated_content_wprompt = {
    "prompt": question,
    "items": generated_content
}

with open(f"{output_filename}.json", 'w') as f:
    json.dump(generated_content_wprompt, f, indent=2)