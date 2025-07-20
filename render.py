import json
from flask import Flask, render_template, jsonify, request
import os

app = Flask(__name__)


def load_items(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data
    

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/llava7_desc')
def render_llava7():
    generated_data = load_items(f"llava7_desc_qna.json")
    if not generated_data: generated_data = {'prompt': 'Error: Could not load data.', 'items': []}
    
    # For initial page load, only send basic item info without heavy content
    item_list = generated_data.get('items', [])
    page_title = generated_data.get('prompt')

    # Sort items by item_id
    item_list.sort(key=lambda item: item['item_id'])

    # Only send item_ids and basic info for initial load
    initial_items = [{'item_id': item.get('item_id')} for item in item_list]

    return render_template(
        'display.html', 
        page_title = page_title,
        item_list = initial_items,
        model_name = f"llava7_desc_qna"
    )


@app.route('/gemma27_<prompt_type>')
def render_gemma27(prompt_type = "desc"):
    generated_data = load_items(f"gemma27_{prompt_type}_qna.json")
    if not generated_data: 
        generated_data = {'prompt': 'Error: Could not load data.', 'items': []}
    
    # For initial page load, only send basic item info without heavy content
    item_list = generated_data.get('items', [])
    page_title = generated_data.get('prompt')

    # Sort items by item_id
    item_list.sort(key=lambda item: item['item_id'])

    # Only send item_ids and basic info for initial load
    initial_items = [{'item_id': item.get('item_id')} for item in item_list]

    return render_template(
        'display.html', 
        page_title = page_title,
        item_list = initial_items,
        model_name = f"gemma27_{prompt_type}"
    )


BEST_OBJECTS = [
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

@app.route('/min_<model>_<prompt_type>')
@app.route('/old_min_gemma27_<prompt_type>', defaults={'model': 'gemma27', 'is_old': True})
def render_model_output(model, prompt_type="desc", is_old=False):
    if is_old:
        filename = f"old_{model}_{prompt_type}_qna.json"
    else:
        filename = f"{model}_{prompt_type}_qna.json"

    generated_data = load_items(filename)
    
    if not generated_data:
        generated_data = {'prompt': f'Error: Could not load data from {filename}', 'items': []}
    
    item_list = generated_data.get('items', [])
    page_title = generated_data.get('prompt')

    # This part of the logic remains the same
    data_root = "media7link/gpt4point_test/"
    image_names = ['000.png', '010.png', '015.png', '020.png']

    best_objects = []

    # Make sure we don't error out if an item is not found
    for object_id in BEST_OBJECTS:
        found_item = next((item for item in item_list if item.get('item_id') == object_id), None)
        if found_item:
            found_item['image_path'] = [os.path.join(data_root, object_id, img_name) for img_name in image_names]
            best_objects.append(found_item)

    best_objects.sort(key=lambda item: item['item_id'])

    return render_template(
        'min.html', 
        page_title=page_title,
        item_list=best_objects
    )


@app.route('/api/item/<item_id>')
def get_item_details(item_id):
    model_name = request.args.get('model', '')
    generated_data = load_items(f"{model_name}_qna.json")
    
    if not generated_data:
        return jsonify({'error': 'Could not load data'}), 500
    
    item_list = generated_data.get('items', [])
    
    # Find the specific item
    item = next((item for item in item_list if item.get('item_id') == item_id), None)
    
    if not item:
        return jsonify({'error': 'Item not found'}), 404
    
    # Add image paths
    data_root = "media7link/gpt4point_test/"
    image_names = ['000.png', '010.png', '015.png', '020.png']
    
    item['image_path'] = [os.path.join(data_root, item_id, img_name) for img_name in image_names]
    
    return jsonify(item)


@app.route('/api/items')
def get_items_batch():
    model_name = request.args.get('model', '')
    item_ids = request.args.getlist('ids')  # Get list of item IDs
    
    generated_data = load_items(f"{model_name}_qna.json")
    
    if not generated_data:
        return jsonify({'error': 'Could not load data'}), 500
    
    item_list = generated_data.get('items', [])
    data_root = "media7link/gpt4point_test/"
    image_names = ['000.png', '010.png', '015.png', '020.png']
    
    # Filter and prepare requested items
    requested_items = []
    for item in item_list:
        if item.get('item_id') in item_ids:
            item['image_path'] = [os.path.join(data_root, item.get('item_id'), img_name) for img_name in image_names]
            requested_items.append(item)
    
    return jsonify(requested_items)


if __name__ == '__main__':
    app.run(host='localhost', port=1337, debug=True)