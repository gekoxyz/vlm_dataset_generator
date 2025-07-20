import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

MODEL_PATH = "/media/data2/mgaliazzo/"
os.environ['HF_HOME'] = MODEL_PATH
os.environ['HF_DATASETS_CACHE'] = MODEL_PATH
os.environ['TRANSFORMERS_CACHE'] = MODEL_PATH

from PIL import Image
import json
from tqdm import tqdm

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig

os.environ['CUDA_HOME'] = '/usr/local/cuda-12.4'
os.environ['PATH'] = f"{os.environ['CUDA_HOME']}/bin:{os.environ['PATH']}"
os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CUDA_HOME']}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

model_id = "llava-hf/llava-onevision-qwen2-72b-ov-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    quantization_config=bnb_config,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)

image_names = ['000.png', '010.png', '015.png', '020.png']
data_root = "media7link/gpt4point_test/"

object_ids = [
    "02797d5feaac4ccabfdf8b357fa2a13a",
    "05068915ce654951910e905e24e35d38",
    "0fa42f5b83084f0eb32533b760c8d146",
    "103989411047470ab9f86341fd016539",
    "18d738a49a2c474281a2675eb35de9b9",
    "1e488ff902e34e62affd7961c88293bb",
    "206b724abdf2486db5e8556853274cb7",
    "3729b2dd716f4c89b87a192290295808",
    "57c8bcfbaa8d4b7d898e74671da510cd",
    "58cd445c1e0044dd8af2009d51b7be18",
    "64ad49b1a1ca425480a28a23dfa151a4",
    "72cd5e73cd9a4e29b11fea522a7ca6bc",
    "73b1f58d52e24b5ca601f54bf33d85c6",
    "7c00eea07b004402ac5b63ace4b2b78f",
    "9ecb745c71f64abfb0faba54a6efb9d0",
    "a903bce6644a4b4692043b3ee1ddbb2b",
    "ba341c4ce89647ea9f6996ec58e3eacf",
    "c55eff0309a14cf09423d238900cc7c2",
    "c5ea812863d746fbab921844294b888a",
    "dec1bb1c2b85451183f33066311e73a8",
    "e205fc3ff5d84b65a4fd89c68af6068e",
    "ee801bec93124d479ef2d41d4592d78a",
    "f699052fd5d7428ca67ba8e84afa1246",
    "f93bb826f374423681a4772a3c49c1df",
    "fb182647a3c447d69944d50e4ccd718e",
    "ff1c458022734dbda358ab2f73a62fa2"
] 

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

for object_id in tqdm(object_ids):
    images_paths = [os.path.join(data_root, object_id, img_name) for img_name in image_names]
    images = [Image.open(p) for p in images_paths]

    basic_description = get_basic_description_by_object_id(gpt4point_basic_descriptions, object_id)

    prompt = f"""You are a meticulous and precise visual analyst. Your task is to generate a single, factual, and objective paragraph describing a scene. The description's primary focus must be the spatial arrangement of the objects within it.

### Core Principles:
1.  Describe, Don't Interpret: Report only what you see. Do not infer actions, intentions, or history. Stick to concrete, observable facts.
2.  Focus on Spatial Relationships: While object attributes like color and shape are important for identification, the paragraph must be structured around *where things are* in relation to one another. Use clear prepositions (e.g. "on top of", "in front of", "next to") without using ambiguous terms such as "to the left/right of".
3. No Speculation: Avoid making assumptions. If you are uncertain about a material, describe its visual properties (e.g. "a dark, textured wood") rather than guessing a specific type (e.g. "oak"). If you cannot identify an object with certainty, describe its shape and color.
4. Literal and Unimaginative: Your goal is to be a camera, not a storyteller. Avoid creating a narrative or setting a mood. Stick to concrete, observable facts.

### Reference Description:
You may be provided with a basic description to identify the main subject(s). Use this to ground your description, but the image is your sole source of truth.
{basic_description}

### Task:
Based on the provided image(s) and the principles above, generate a single, detailed paragraph. The paragraph should identify the Primary Objects, their key visual attributes for identification, and, most importantly, their spatial layout and relationships to each other.
Please analyze the scene using the given instructions."""

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

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    decoded = processor.decode(generation[0], skip_special_tokens=True)
    model_response = decoded.split('Please analyze the scene using the given instructions.assistant\n',1)[1]

    item_data = {
        "item_id": object_id,
        "basic_description": basic_description,
        "augmented_description": model_response
    }
    generated_content.append(item_data)

output_filename = "llava72_desc_NEW"

basic_description = "BASIC_DESCRIPTION"
generated_content_wprompt = {
    "prompt": [f"{prompt}\n{question}"],
    "items": generated_content
}

with open(f"{output_filename}.json", 'w') as f:
    json.dump(generated_content_wprompt, f, indent=2)