import argparse

import wandb

api = wandb.Api()


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project-name", type=str, default="cleanrl",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="openrlbenchmark",
        help="the entity (team) of wandb's project")

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
    print(args)
    runs = api.runs(path=f"{args.wandb_entity}/{args.wandb_project_name}", filters={"tags": {"$in": [args.source_tag]}})
    for run in runs:
        tags = run.tags
        if args.add:
            tags.append(args.add)
        if args.remove:
            tags.remove(args.remove)
        run.tags = tags
        run.update()
