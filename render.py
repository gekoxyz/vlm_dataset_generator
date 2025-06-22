import json
from flask import Flask, render_template
import os
import re

# Initialize the Flask application
app = Flask(__name__)

def load_items(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print("ERROR: data.json not found. Make sure the file exists.")
        return None
    except json.JSONDecodeError:
        print("ERROR: data.json is not valid JSON. Please check the file format.")
        return None


@app.route('/')
@app.route('/<model_name>')
def render_page(model_name: str = 'llava7'):
    generated_data = load_items(f"{model_name}_decomposition_qna.json")
    if not generated_data: generated_data = {'prompt': 'Error: Could not load data.', 'items': []}
    
    item_list = generated_data.get('items', [])
    page_title = generated_data.get('prompt')

    data_root = "media7link/gpt4point_test/"
    image_names = ['000.png', '010.png', '015.png', '020.png']

    # Loop through each item in the list and add its image path
    for item in item_list:
        item_id = item.get('item_id')
        if item_id:
            item['image_path'] = [os.path.join(data_root, item_id ,img_name) for img_name in image_names]
        else:
            item['image_path'] = ''

    item_list.sort(key=lambda item: item['item_id'])

    # Unpack the data and pass it to the template with specific names
    return render_template(
        'index.html', 
        page_title=page_title,
        item_list=item_list
    )


@app.route('/differences')
def render_differences():
    with open("desc_differences.json", 'r') as f:
        data = json.load(f)

    data_root = "media7link/gpt4point_test/"
    image_names = ['000.png', '010.png', '015.png', '020.png']

    # Loop through each item in the list and add its image path
    for item in data:
        item_id = item.get('item_id')
        if item_id:
            item['image_path'] = [os.path.join(data_root, item_id ,img_name) for img_name in image_names]
        else:
            item['image_path'] = ''
        
        item_diffs = item.get('differences')

        pattern = re.compile(r"Description 1 with Highlights:\s*(.*?)\s*Description 2 with Highlights:\s*(.*)")
        match = pattern.search(item_diffs)

        if match:
            between_patterns = match.group(1)
            after_pattern2 = match.group(2)

            item['desc_1'] = between_patterns
            item['desc_2'] = after_pattern2
        else:
            item['desc_1'] = "PATTERN NOT FOUND IN THE STRING"
            item['desc_2'] = "PATTERN NOT FOUND IN THE STRING"


    # Unpack the data and pass it to the template with specific names
    return render_template(
        'differences.html', 
        item_list=data
    )



if __name__ == '__main__':
    app.run(host='localhost', port=1337, debug=True)
