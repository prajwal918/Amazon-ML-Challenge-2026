import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import kaggle
from kaggle.models.api_list_kernel_session_output_request import ApiListKernelSessionOutputRequest
import requests

api = kaggle.KaggleApi()
api.authenticate()

kernels = [
    ("node_beta", "amazon-ml-challenge-v3-resolver", "v7"),
    ("node_beta", "amazon-ml-challenge-v8-dp-resolver", "v8"),
    ("node_beta", "amazon-ml-challenge-v9-precision-dp", "v9"),
    ("node_beta", "amazon-ml-challenge-v10-phonetic", "v10"),
]

for owner, slug, ver in kernels:
    print("=" * 70)
    print(f"Kernel: {owner}/{slug} ({ver})")
    print("=" * 70)
    
    with api.build_kaggle_client() as client:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner
        req.kernel_slug = slug
        try:
            resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
            files = [f.file_name for f in (resp.files or [])]
            print("Files available on Kaggle:", files)
            
            # Print last 500 chars of log safely
            log = resp.log or ""
            print(f"Log length: {len(log):,} chars")
            print("LOG TAIL:")
            print(log[-1500:])
            
            # If files exist, download them manually without cp1252 crash
            target_dir = f"safe_output_{ver}"
            os.makedirs(target_dir, exist_ok=True)
            for item in resp.files or []:
                out_path = os.path.join(target_dir, item.file_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                print(f"Downloading {item.file_name}...")
                r = requests.get(item.url, stream=True)
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
                print(f"  -> Saved {out_path} ({os.path.getsize(out_path):,} bytes)")
                
            # Save log in utf-8
            with open(os.path.join(target_dir, f"{slug}.log"), "w", encoding="utf-8") as f:
                f.write(log)
                
        except Exception as e:
            print("Error inspecting kernel:", e)
    print()
