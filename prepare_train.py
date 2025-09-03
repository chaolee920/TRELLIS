import json
import os
from huggingface_hub import hf_hub_download

# Download a sample JSON file
repo_id = "404-Gen/404mini"
filename = "assets/an/antique_toy_train_set_running.json"
local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

# Read and print the content
with open(local_path, 'r') as f:
    data = json.load(f)
print(data)