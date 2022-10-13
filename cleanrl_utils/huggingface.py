import argparse
import os
from typing import List
import numpy as np
from pathlib import Path
from pprint import pformat

HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME = "replay.mp4"

def upload_to_hub(
    args: argparse.Namespace,
    episodic_returns: List,
    repo_id: str,
    algo_name: str,
    folder_path: str,
    video_folder_path: str = "",
):
    # Step 1: lazy import and create / read a huggingface repo
    from huggingface_hub import HfApi, upload_folder, Repository
    from huggingface_hub.repocard import metadata_eval_result, metadata_save

    api = HfApi()
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
    )
    # parse the default entity
    entity, repo = repo_url.split("/")[-2:]
    repo_id = f"{entity}/{repo}"

    # Step 2: clean up data
    # delete previous tfevents and mp4 files
    for file in api.list_repo_files(repo_id=repo_id):
        if ".tfevents" in file or ".mp4" in file:
            api.delete_file(file, repo_id=repo_id)
    # fetch mp4 files
    if video_folder_path:
        video_files = [str(item) for item in Path(video_folder_path).glob("*.mp4")]
        print(video_files)
        # sort by the number in the file name
        video_files = sorted(video_files, key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))
        for file in video_files:
            api.upload_file(path_or_fileobj=file, path_in_repo=file, repo_id=repo_id)
        api.upload_file(path_or_fileobj=file, path_in_repo=HUGGINGFACE_VIDEO_PREVIEW_FILE_NAME, repo_id=repo_id)

    # Step 3: Generate the model card
    model_card = f"""
# (CleanRL) **{algo_name}** Agent Playing **{args.env_id}**

This is a trained model of a {algo_name} agent playing {args.env_id}.
The model was trained by using [CleanRL](https://github.com/vwxyzjn/cleanrl) and the training code can be
found [here](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/{args.exp_name}.py).


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
    repo_url = upload_folder(
        repo_id=repo_id,
        folder_path=folder_path,
        path_in_repo="",
        commit_message="pushing model",
    )
    return repo_url