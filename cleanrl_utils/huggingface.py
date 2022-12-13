import argparse
import os
import sys
from pathlib import Path
from pprint import pformat
from typing import List

import numpy as np

HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME = "replay.mp4"
HUGGINGFACE_README_FILE_NAME = "README.md"


def push_to_hub(
    args: argparse.Namespace,
    episodic_returns: List,
    repo_id: str,
    algo_name: str,
    folder_path: str,
    video_folder_path: str = "",
    revision: str = "main",
    create_pr: bool = False,
    private: bool = False,
):
    # Step 1: lazy import and create / read a huggingface repo
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    from huggingface_hub.repocard import metadata_eval_result, metadata_save

    api = HfApi()
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
        private=private,
    )
    # parse the default entity
    entity, repo = repo_url.split("/")[-2:]
    repo_id = f"{entity}/{repo}"

    # Step 2: clean up data
    # delete previous tfevents and mp4 files
    operations = [
        CommitOperationDelete(path_in_repo=file)
        for file in api.list_repo_files(repo_id=repo_id)
        if file.endswith(".tfevents") or file.endswith(".mp4")
    ]

    # Step 3: Generate the model card
    algorith_variant_filename = sys.argv[0].split("/")[-1]
    model_card = f"""
# (CleanRL) **{algo_name}** Agent Playing **{args.env_id}**

This is a trained model of a {algo_name} agent playing {args.env_id}.
The model was trained by using [CleanRL](https://github.com/vwxyzjn/cleanrl) and the most up-to-date training code can be
found [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/{args.exp_name}.py).

## Command to reproduce the training

```bash
curl -OL https://huggingface.co/{repo_id}/raw/main/dqn.py
curl -OL https://huggingface.co/{repo_id}/raw/main/pyproject.toml
curl -OL https://huggingface.co/{repo_id}/raw/main/poetry.lock
poetry install --all-extras
python {algorith_variant_filename} {" ".join(sys.argv[1:])}
```

# Hyperparameters
```python
{pformat(vars(args))}
```
    """
    readme_path = f"{folder_path}/README.md"
    readme = model_card

    # metadata
    metadata = {}
    metadata["tags"] = [
        args.env_id,
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "custom-implementation",
    ]
    metadata["library_name"] = "cleanrl"
    eval = metadata_eval_result(
        model_pretty_name=algo_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{np.average(episodic_returns):.2f} +/- {np.std(episodic_returns):.2f}",
        dataset_pretty_name=args.env_id,
        dataset_id=args.env_id,
    )
    metadata = {**metadata, **eval}

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    metadata_save(readme_path, metadata)

    # fetch mp4 files
    if video_folder_path:
        video_files = [str(item) for item in Path(video_folder_path).glob("*.mp4")]
        # sort by the number in the file name
        video_files = sorted(video_files, key=lambda x: int("".join(filter(str.isdigit, os.path.splitext(x)[0]))))
        for file in video_files:
            operations += [CommitOperationAdd(path_or_fileobj=file, path_in_repo=file)]
        operations += [CommitOperationAdd(path_or_fileobj=file, path_in_repo=HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME)]

    # fetch folder files
    for item in [str(item) for item in Path(folder_path).glob("*")]:
        operations += [CommitOperationAdd(path_or_fileobj=item, path_in_repo=os.path.relpath(item, folder_path))]

    # fetch source code
    operations += [CommitOperationAdd(path_or_fileobj=sys.argv[0], path_in_repo=sys.argv[0].split("/")[-1])]

    # upload poetry files at the root of the repository
    git_root = Path(__file__).parent.parent
    operations += [CommitOperationAdd(path_or_fileobj=f"{git_root}/pyproject.toml", path_in_repo=f"pyproject.toml")]
    operations += [CommitOperationAdd(path_or_fileobj=f"{git_root}/poetry.lock", path_in_repo=f"poetry.lock")]

    api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message="pushing model",
        revision=revision,
        create_pr=create_pr,
    )
    print(f"Model pushed to {repo_url}")
    return repo_url
