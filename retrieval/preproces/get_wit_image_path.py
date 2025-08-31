import os
import pandas as pd
import json

folder_path = "/root/autodl-tmp/my_images1"

output_tsv_path = "wit_image_path.tsv"

data = []

for root, dirs, files in os.walk(folder_path):
    # print(f"Processing {root}")
    if root == folder_path:
        files = []
    for file_name in files:
        # print(f"Processing {file_name}")
        if file_name.endswith(".json"):
            json_file_path = os.path.join(root, file_name)
            
            # print(f"Processing {json_file_path}")

            with open(json_file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            
            try:
                image_file_name = json_data["key"] + ".jpg"
            except KeyError:
                print(f"Key not found in {json_file_path}")
                exit(1)
            image_absolute_path = os.path.abspath(os.path.join(root, image_file_name))  
            # image_relative_path = os.path.join(".", image_file_name) 
            data.append({
                "caption": json_data["caption"],
                "url": json_data["url"],
                "width": json_data["width"],
                "height": json_data["height"],
                "   ": image_absolute_path,  
                # "relative_image_path": image_relative_path  
            })

df = pd.DataFrame(data)
df.to_csv(output_tsv_path, sep="\t", index=False)

print(f"Data saved to {output_tsv_path}")