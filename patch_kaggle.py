import os

path = r"~/site-packages\kaggle\api\kaggle_api_extended.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = 'with open(outfile, "w") as out:'
replacement = 'with open(outfile, "w", encoding="utf-8") as out:'

if target in content:
    content = content.replace(target, replacement)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: Patched kaggle_api_extended.py with utf-8 encoding!")
else:
    print("Target not found.")
