import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

MODEL_PATH = "/media/data2/mgaliazzo/"
os.environ['HF_HOME'] = MODEL_PATH
os.environ['HF_DATASETS_CACHE'] = MODEL_PATH
os.environ['TRANSFORMERS_CACHE'] = MODEL_PATH

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import copy
import torch
from tqdm import tqdm
import json

pretrained = "lmms-lab/llava-critic-72b"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(
    pretrained, 
    None, 
    model_name, 
    load_4bit=True,
    device_map=device_map
)

model.eval()

image_names = ['000.png', '010.png', '015.png', '020.png']
data_root = "media7link/gpt4point_test/"

object_ids = [
    "02797d5feaac4ccabfdf8b357fa2a13a",
    # "05068915ce654951910e905e24e35d38",
    # "0fa42f5b83084f0eb32533b760c8d146",
    # "103989411047470ab9f86341fd016539",
    # "18d738a49a2c474281a2675eb35de9b9",
    # "1e488ff902e34e62affd7961c88293bb",
    # "206b724abdf2486db5e8556853274cb7",
    # "3729b2dd716f4c89b87a192290295808",
    # "57c8bcfbaa8d4b7d898e74671da510cd",
    # "58cd445c1e0044dd8af2009d51b7be18",
    # "64ad49b1a1ca425480a28a23dfa151a4",
    # "72cd5e73cd9a4e29b11fea522a7ca6bc",
    # "73b1f58d52e24b5ca601f54bf33d85c6",
    # "7c00eea07b004402ac5b63ace4b2b78f",
    # "9ecb745c71f64abfb0faba54a6efb9d0",
    # "a903bce6644a4b4692043b3ee1ddbb2b",
    # "ba341c4ce89647ea9f6996ec58e3eacf",
    # "c55eff0309a14cf09423d238900cc7c2",
    # "c5ea812863d746fbab921844294b888a",
    # "dec1bb1c2b85451183f33066311e73a8",
    # "e205fc3ff5d84b65a4fd89c68af6068e",
    # "ee801bec93124d479ef2d41d4592d78a",
    # "f699052fd5d7428ca67ba8e84afa1246",
    # "f93bb826f374423681a4772a3c49c1df",
    # "fb182647a3c447d69944d50e4ccd718e",
    # "ff1c458022734dbda358ab2f73a62fa2"
] 

llava_prompt = """You are a meticulous and precise visual analyst. Your task is to generate a single, factual, and objective paragraph describing a scene. The description's primary focus must be the spatial arrangement of the objects within it.

### Core Principles:
1.  Describe, Don't Interpret: Report only what you see. Do not infer actions, intentions, or history. Stick to concrete, observable facts.
2.  Focus on Spatial Relationships: While object attributes like color and shape are important for identification, the paragraph must be structured around *where things are* in relation to one another. Use clear prepositions (e.g. "on top of", "in front of", "next to") without using ambiguous terms such as "to the left/right of".
3. No Speculation: Avoid making assumptions. If you are uncertain about a material, describe its visual properties (e.g. "a dark, textured wood") rather than guessing a specific type (e.g. "oak"). If you cannot identify an object with certainty, describe its shape and color.
4. Literal and Unimaginative: Your goal is to be a camera, not a storyteller. Avoid creating a narrative or setting a mood. Stick to concrete, observable facts.

### Task:
Based on the provided image(s) and the principles above, generate a single, detailed paragraph. The paragraph should identify the Primary Objects, their key visual attributes for identification, and, most importantly, their spatial layout and relationships to each other.
Please analyze the scene using the given instructions."""

llava_response = ""

DESCRIPTIONS_JSON_PATH = "llava72_desc_qna"
def load_json(file_path):
    with open(file_path, 'r') as file: return json.load(file)

descriptions = load_json(f"{DESCRIPTIONS_JSON_PATH}.json").get("items", [])

for item in tqdm(descriptions):
    images_paths = [os.path.join(data_root, item.get("item_id", ""), img_name) for img_name in image_names]

    images = []
    for image_path in images_paths:
        image = Image.open(image_path).convert("RGB")
        images.append(image)
    
    image_tensor = process_images(images, image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

    llava_response = item.get("augmented_description", "")

    # pointwise scoring
    critic_prompt = f"Given an image and a corresponding question, please serve as an unbiased and fair judge to evaluate the quality of answer answers provided by a Large Multimodal Model (LMM). Score the response out of 100 and explain your reasoning with specific details.\nQuestion: {llava_prompt}\nThe LMM response: {llava_response}\n. Please evaluate the quality of this answer."

    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(images)
    question = "".join(image_tokens) + "\n" + critic_prompt

    # question = DEFAULT_IMAGE_TOKEN + "\n" + critic_prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [img.size for img in images]
    
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    print(text_outputs[0])
    print(f"output len: {len(text_outputs)}")

    item["critic"] = text_outputs[0]


# --- Save Results ---
prompts = load_json(f"{DESCRIPTIONS_JSON_PATH}.json").get("prompt", [])
prompts.append(critic_prompt)
generated_content_clean_qna = {"prompt" : prompts, "items": descriptions}

with open(f"{DESCRIPTIONS_JSON_PATH}_critic.json", 'w') as f: json.dump(generated_content_clean_qna, f, indent=2)

print("Processing complete. Output saved.")