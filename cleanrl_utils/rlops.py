import argparse
from distutils.util import strtobool
from typing import Dict, List

import expt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import wandb.apis.reports as wb  # noqa
from expt import Hypothesis, Run
from rich.console import Console

wandb.require("report-editing")
api = wandb.Api()


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ddpg_continuous_action_jax",
        help="the name of this experiment")
    parser.add_argument("--wandb-project-name", type=str, default="cleanrl",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="openrlbenchmark",
        help="the entity (team) of wandb's project")
    parser.add_argument("--tags", nargs="+", default=["v1.0.0b2-9-g4605546", "rlops-pilot"],
        help='the tags of the runsets (e.g., `--tags v1.0.0b2-9-g4605546 rlops-pilot` and you can also use `--tags "v1.0.0b2-9-g4605546;latest"` to filter runs with multiple tags)')
    parser.add_argument("--env-ids", nargs="+", default=["Hopper-v2", "Walker2d-v2", "HalfCheetah-v2"],
        help="the ids of the environment to compare")
    parser.add_argument("--output-filename", type=str, default="compare.png",
        help="the output filename of the plot")
    parser.add_argument("--rolling", type=int, default=100,
        help="the rolling window for smoothing the curves")
    parser.add_argument("--metric-last-n-average-window", type=int, default=100,
        help="the last n number of episodes to average metric over in the result table")
    parser.add_argument("--scan-history", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, we will pull the complete metrics from wandb instead of sampling 500 data points (recommended for generating tables)")
    parser.add_argument("--report", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, a wandb report will be created")
    # fmt: on
    return parser.parse_args()


def create_hypothesis(name: str, wandb_runs: List[wandb.apis.public.Run], scan_history: bool = False) -> Hypothesis:
    runs = []
    for idx, run in enumerate(wandb_runs):
        if scan_history:
            wandb_run = pd.DataFrame([row for row in run.scan_history()])
        else:
            wandb_run = run.history()
        if "videos" in wandb_run:
            wandb_run = wandb_run.drop(columns=["videos"], axis=1)
        runs += [Run(f"seed{idx}", wandb_run)]
    return Hypothesis(name, runs)


class Runset:
    def __init__(self, name: str, filters: dict, entity: str, project: str, groupby: str = ""):
        self.name = name
        self.filters = filters
        self.entity = entity
        self.project = project
        self.groupby = groupby

    @property
    def runs(self):
        return wandb.Api().runs(path=f"{self.entity}/{self.project}", filters=self.filters)

    @property
    def report_runset(self):
        return wb.RunSet(
            name=self.name,
            entity=self.entity,
            project=self.project,
            filters={"$or": [self.filters]},
            groupby=[self.groupby] if len(self.groupby) > 0 else None,
        )


def compare(
    runsetss: List[List[Runset]],
    env_ids: List[str],
    ncols: int,
    rolling: int,
    metric_last_n_average_window: int,
    scan_history: bool = False,
    output_filename: str = "compare.png",
):
    blocks = []
    for idx, env_id in enumerate(env_ids):
        blocks += [
            wb.PanelGrid(
                runsets=[runsets[idx].report_runset for runsets in runsetss],
                panels=[
                    wb.LinePlot(
                        x="global_step",
                        y=["charts/episodic_return"],
                        title=env_id,
                        title_x="Steps",
                        title_y="Episodic Return",
                        max_runs_to_show=100,
                        smoothing_factor=0.8,
                        groupby_rangefunc="stderr",
                        legend_template="${runsetName}",
                    ),
                    wb.LinePlot(
                        x="_runtime",
                        y=["charts/episodic_return"],
                        title=env_id,
                        title_y="Episodic Return",
                        max_runs_to_show=100,
                        smoothing_factor=0.8,
                        groupby_rangefunc="stderr",
                        legend_template="${runsetName}",
                    ),
                    # wb.MediaBrowser(
                    #     num_columns=2,
                    #     media_keys="videos",
                    # ),
                ],
            ),
        ]

    nrows = np.ceil(len(env_ids) / ncols).astype(int)
    figsize = (ncols * 4, nrows * 3)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        # sharex=True,
        # sharey=True,
    )

    result_table = pd.DataFrame(index=env_ids, columns=[runsets[0].name for runsets in runsetss])
    for idx, env_id in enumerate(env_ids):
        ex = expt.Experiment("Comparison")
        for runsets in runsetss:
            h = create_hypothesis(runsets[idx].name, runsets[idx].runs, scan_history)
            ex.add_hypothesis(h)

        # for each run `i` get the average of the last `rolling` episodes as r_i
        # then take the average and std of r_i as the results.
        result = []
        for hypothesis in ex.hypotheses:
            raw_result = []
            for run in hypothesis.runs:
                raw_result += [run.df["charts/episodic_return"].dropna()[-metric_last_n_average_window:].mean()]
            raw_result = np.array(raw_result)
            result += [f"{raw_result.mean():.2f} ± {raw_result.std():.2f}"]
        result_table.loc[env_id] = result

        ax = axes.flatten()[idx]
        ex.plot(
            ax=ax,
            title=env_id,
            x="global_step",
            y="charts/episodic_return",
            err_style="band",
            std_alpha=0.1,
            rolling=rolling,
            # n_samples=500,
            legend=False,
        )

    print(result_table)

    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2)
    fig.subplots_adjust(top=0.9)
    # remove the empty axes
    for ax in axes.flatten()[len(env_ids) :]:
        ax.remove()

    print(f"saving figure to {output_filename}")
    plt.savefig(f"{output_filename}", bbox_inches="tight")
    plt.savefig(f"{output_filename.replace('.png', '.pdf')}", bbox_inches="tight")
    return blocks


def parse_str(tag_str: str) -> List[List[Dict]]:
    """
    Extract the user, include tags, and exclude tags from the tag string `tag_str`.

    `tag_str` has the syntax "tag1;tag2!tag3;tag4?user1", where the tag1 and tag2 are the
    include tags, tag3 and tag4 are the exclude tags, and user1 is the user

    :param tag_str: a string of the form
    :return: a list of filters for wandb
    """
    user = []
    if "?" in tag_str:
        user = tag_str.split("?")[1]
        tag_str = tag_str.split("?")[0]
    # extract exclude tags
    exclude_tag_groups = []
    if "!" in tag_str:
        exclude_tag_groups = tag_str.split("!")[1].split(";")
        tag_str = tag_str.split("!")[0]
    # extract include tags
    include_tag_groups = tag_str.split(";")

    # convert to wandb filters
    include_tag_groups = [{"tags": {"$in": [tag]}} for tag in include_tag_groups]
    exclude_tag_groups = [{"tags": {"$nin": [tag]}} for tag in exclude_tag_groups]
    user = [{"username": user}] if len(user) > 0 else []

    return include_tag_groups, exclude_tag_groups, user


if __name__ == "__main__":
    args = parse_args()
    console = Console()
    blocks = []
    runsetss = []
    for tag_str in args.tags:
        include_tag_groups, exclude_tag_groups, user = parse_str(tag_str)
        runsets = []
        for env_id in args.env_ids:
            runsets += [
                Runset(
                    name=f"CleanRL's {args.exp_name} ({tag_str})",
                    filters={
                        "$and": [
                            {"config.env_id.value": env_id},
                            *include_tag_groups,
                            *exclude_tag_groups,
                            *user,
                            {"config.exp_name.value": args.exp_name},
                        ]
                    },
                    entity=args.wandb_entity,
                    project=args.wandb_project_name,
                    groupby="exp_name",
                )
            ]
            console.print(
                f"CleanRL's {args.exp_name} [green]({tag_str})[/] in [purple]{env_id}[/] has {len(runsets[-1].runs)} runs"
            )
            for run in runsets[-1].runs:
                console.print(f"┣━━ [link={run.url}]{run.name}[/link] with tags = {run.tags}")
            assert len(runsets[0].runs) > 0, f"CleanRL's {args.exp_name} ({tag_str}) in {env_id} has no runs"
        runsetss += [runsets]

    blocks = compare(
        runsetss,
        args.env_ids,
        output_filename="compare.png",
        ncols=2,
        rolling=args.rolling,
        metric_last_n_average_window=args.metric_last_n_average_window,
        scan_history=args.scan_history,
    )
    if args.report:
        print("saving report")
        report = wb.Report(
            project="cleanrl",
            title=f"Regression Report: {args.exp_name} ({args.tags})",
            blocks=blocks,
        )
        report.save()
        print(f"view the generated report at {report.url}")
