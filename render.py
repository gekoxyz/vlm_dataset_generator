import json
from flask import Flask, render_template
import os

# Initialize the Flask application
app = Flask(__name__)

JSON_DATA = 'llava7_desc_qna.json'

def load_items():
    try:
        with open(JSON_DATA, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print("ERROR: data.json not found. Make sure the file exists.")
        return None
    except json.JSONDecodeError:
        print("ERROR: data.json is not valid JSON. Please check the file format.")
        return None

@app.route('/')
def render():
    generated_data = load_items()
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

    # Unpack the data and pass it to the template with specific names
    return render_template(
        'index.html', 
        page_title=page_title,
        item_list=item_list
    )

# This is the standard entry point for a Python script.
# The server will only run if the script is executed directly.
if __name__ == '__main__':
    # app.run() starts the development server.
    # host='localhost' makes it accessible only on your computer.
    # port=1337 is the specific port you requested.
    # debug=True provides helpful error messages and auto-reloads the server
    # when you make changes to the code.
    app.run(host='localhost', port=1337, debug=True)