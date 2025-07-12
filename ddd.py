import json

def load_items(json_path):
    """
    Loads JSON data from a specified file path.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data

def save_items(data, json_path):
    """
    Saves JSON data to a specified file path.
    """
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4) # Using indent for pretty printing

# Load the JSON data
json_file_path = 'gemma27_deco_qna.json'
llava_json = load_items(json_file_path)

# Get the list of items
item_list = llava_json.get('items', [])

# Iterate through each item and remove the 'generated_qnas' field
for item in item_list:
    if 'generated_qnas' in item:
        del item['generated_qnas']
        print(f"Removed 'generated_qnas' from item with ID: {item.get('id', 'N/A')}")
    else:
        print(f"Item with ID: {item.get('id', 'N/A')} does not have 'generated_qnas' field.")

# Optionally, save the modified data back to a new JSON file
# It's good practice to save to a new file to keep the original untouched.
output_json_file_path = 'gemma27_deco.json'
save_items(llava_json, output_json_file_path)

print(f"\nModified data saved to: {output_json_file_path}")

# You can also verify by loading the new file and checking an item
# modified_data_check = load_items(output_json_file_path)
# if modified_data_check.get('items'):
#     print("\nFirst item in modified data (check for 'generated_qnas'):")
#     print(json.dumps(modified_data_check['items'][0], indent=2))