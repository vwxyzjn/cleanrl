import argparse
import json
import requests
from distutils.util import strtobool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CleanRL Plots')
    # Common arguments
    parser.add_argument('--run', type=str, default="cleanrl/cleanrl.benchmark/runs/thq5rgnz",
                        help='the name of wandb project (e.g. cleanrl/cleanrl)')
    parser.add_argument('--remove-entity', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, the wandb-entity will be removed')
    args = parser.parse_args()
    uri = args.run.replace("/runs", "")

    requirements_txt_url = f"https://api.wandb.ai/files/{uri}/requirements.txt"
    metadata_url = f"https://api.wandb.ai/files/{uri}/wandb-metadata.json"
    metadata = requests.get(url=metadata_url).json()
    
    if args.remove_entity:
        a = []
        wandb_entity_idx = None
        for i in range(len(metadata["args"])):
            if metadata["args"][i] == "--wandb-entity":
                wandb_entity_idx = i
                continue
            if wandb_entity_idx and i == wandb_entity_idx+1:
                continue
            a += [metadata["args"][i]]
    else:
        a = metadata["args"]

    program = ["python"] + [metadata["program"]] + a


    print(f"""
# run the following
python3 -m venv venv
source venv/bin/activate
pip install -r {requirements_txt_url}
curl -OL https://api.wandb.ai/files/{uri}/code/{metadata["codePath"]}
{" ".join(program)}
""")