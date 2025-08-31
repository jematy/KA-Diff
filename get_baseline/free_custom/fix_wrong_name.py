import os
import re

base_dir = "freecustom/free_custom_dreambooth_42"

for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]

    numbered_files = []
    for f in files:
        match = re.match(r"(\d+)_", f)
        if match:
            num = int(match.group(1))
            numbered_files.append((num, f))

    numbered_files.sort()

    for i, (_, old_name) in enumerate(numbered_files, 1):
        old_path = os.path.join(subfolder_path, old_name)

        suffix = old_name.split("_", 1)[1]
        new_prefix = f"{i:03d}"
        new_name = f"{new_prefix}_{suffix}"
        new_path = os.path.join(subfolder_path, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")
