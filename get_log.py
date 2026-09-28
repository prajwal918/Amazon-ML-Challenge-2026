import os, sys, io

# Force UTF-8 everywhere
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Monkey-patch open to use UTF-8
original_open = open
def utf8_open(*args, **kwargs):
    if len(args) >= 2 and 'w' in str(args[1]):
        kwargs.setdefault('encoding', 'utf-8')
    return original_open(*args, **kwargs)

import builtins
builtins.open = utf8_open

os.makedirs('kaggle_output_v5', exist_ok=True)

import kaggle
api = kaggle.KaggleApi()
api.authenticate()

# This is the correct method
api.kernels_output('node_beta/amazon-ml-challenge-v3-resolver', 'kaggle_output_v5')
print("Download complete!")
