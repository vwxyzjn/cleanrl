import argparse

import wandb
from rich.console import Console

from rlops import parse_str
api = wandb.Api()


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project-name", type=str, default="cleanrl",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="openrlbenchmark",
        help="the entity (team) of wandb's project")
    parser.add_argument("--exp-name", type=str, default="ddpg_continuous_action_jax",
        help="the name of this experiment")
    parser.add_argument("--add", type=str, default="", 
        help="the tag to be added to any runs with the `--source-tag`")
    parser.add_argument("--remove", type=str, default="", 
        help="the tag to be removed from any runs with the `--source-tag`")
    parser.add_argument("--source-tag", type=str, default="v1.0.0b2-7-g4bb6766",
        help="the source tag of the set of runs")
    # fmt: on
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    console = Console()
    include_tag_groups, exclude_tag_groups, user = parse_str(args.source_tag)
    runs = api.runs(path=f"{args.wandb_entity}/{args.wandb_project_name}", filters={
        "$and": [
            *include_tag_groups,
            *exclude_tag_groups,
            *user,
            {"config.exp_name.value": args.exp_name},
        ]
    })
    confirmation_str = "You are about to make the following changes:\n"
    modified_runs = []
    for run in runs:
        tags = run.tags
        if args.add and args.add not in tags:
                confirmation_str += f"Adding the tag '{args.add}' to [link={run.url}]{run.name}[/link], which has tags {str(tags)}\n"
                tags.append(args.add)
                run.tags = tags
                modified_runs.append(run)
        if args.remove and args.remove in tags:
                confirmation_str += f"Removing the tag '{args.remove}' from [link={run.url}]{run.name}[/link], which has tags {str(tags)}\n"
                tags.remove(args.remove)
                run.tags = tags
                modified_runs.append(run)

    console.print(confirmation_str)
    response = input("Are you sure you want to proceed? (y/n):")
    if response.lower() == "y":
        for run in modified_runs:
            print(f"Updating {run.name}")
            run.update()
