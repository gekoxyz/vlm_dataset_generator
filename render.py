import json
from flask import Flask, render_template, jsonify, request
import os

app = Flask(__name__)


def load_items(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data


@app.route('/gemma27_desc')
def render_gemma27():
    generated_data = load_items(f"gemma27_desc_qna.json")
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
        'index.html', 
        page_title=page_title,
        item_list=initial_items,
        model_name="gemma27_desc"
    )


@app.route('/gemma27_deco')
def render_gemma27_deco():
    generated_data = load_items(f"gemma27_deco_qna.json")
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
        'index.html', 
        page_title=page_title,
        item_list=initial_items,
        model_name="gemma27_deco"
    )


@app.route('/')
@app.route('/<model_name>')
def render_page(model_name: str = 'llava7'):
    generated_data = load_items(f"{model_name}_qna.json")
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
        'index.html', 
        page_title=page_title,
        item_list=initial_items,
        model_name=model_name
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