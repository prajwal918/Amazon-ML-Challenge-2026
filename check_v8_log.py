import os, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

original_open = open
def utf8_open(*args, **kwargs):
    if len(args) >= 2 and 'w' in str(args[1]) and 'b' not in str(args[1]):
        kwargs.setdefault('encoding', 'utf-8')
    return original_open(*args, **kwargs)

import builtins
builtins.open = utf8_open

import kaggle
api = kaggle.KaggleApi()
api.authenticate()

os.makedirs('log_v8', exist_ok=True)
print("Fetching V8 kernel output...")
api.kernels_output('node_beta/amazon-ml-challenge-v8-dp-resolver', 'log_v8', force=True)

print("Downloaded files in log_v8:")
for root, dirs, files in os.walk('log_v8'):
    for f in files:
        fp = os.path.join(root, f)
        print(f"  {fp} ({os.path.getsize(fp):,} bytes)")
        if f.endswith('.log'):
            with open(fp, 'r', encoding='utf-8', errors='ignore') as lf:
                lines = lf.readlines()
                print("--- LOG TAIL (LAST 30 LINES) ---")
                print("".join(lines[-30:]))
