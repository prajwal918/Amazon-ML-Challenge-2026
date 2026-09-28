import os, sys, io

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Monkey-patch open to use UTF-8 only for text writing
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

kernels = [
    ('node_beta/amazon-ml-challenge-v3-resolver', 'v7'),
    ('node_beta/amazon-ml-challenge-v8-dp-resolver', 'v8'),
    ('node_beta/amazon-ml-challenge-v9-precision-dp', 'v9'),
    ('node_beta/amazon-ml-challenge-v10-phonetic', 'v10'),
]

for slug, ver in kernels:
    print("=" * 70)
    print(f"Downloading output for {slug} ({ver})...")
    print("=" * 70)
    dest_dir = f"kaggle_download_{ver}"
    os.makedirs(dest_dir, exist_ok=True)
    try:
        api.kernels_output(slug, dest_dir, force=True)
        print("Downloaded files:", os.listdir(dest_dir))
        for f in os.listdir(dest_dir):
            fp = os.path.join(dest_dir, f)
            if os.path.isfile(fp):
                print(f"  {f}: {os.path.getsize(fp):,} bytes")
            elif os.path.isdir(fp):
                print(f"  [{f}/]:", os.listdir(fp))
                for subf in os.listdir(fp):
                    subfp = os.path.join(fp, subf)
                    if os.path.isfile(subfp):
                        print(f"    {subf}: {os.path.getsize(subfp):,} bytes")
    except Exception as e:
        print(f"Error downloading {slug}: {e}")
    print()
