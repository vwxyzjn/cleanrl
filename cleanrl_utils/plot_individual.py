import argparse
import os
import pickle
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb

parser = argparse.ArgumentParser(description="CleanRL Plots")
# Common arguments
parser.add_argument(
    "--wandb-project",
    type=str,
    default="anonymous-rl-code/action-guidance",
    help="the name of wandb project (e.g. cleanrl/cleanrl)",
)
parser.add_argument(
    "--feature-of-interest",
    type=str,
    default="charts/episodic_return/ProduceCombatUnitRewardFunction",
    help="which feature to be plotted on the y-axis",
)
parser.add_argument("--hyper-params-tuned", nargs="+", default=["shift", "adaptation"], help="the hyper parameters tuned")
# parser.add_argument('--scan-history', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
#                     help='if toggled, cuda will not be enabled by default')
parser.add_argument(
    "--interested-exp-names",
    nargs="+",
    default=[
        "ppo_ac_positive_reward-shift-800000--adaptation-1000000--positive_likelihood-1-",
        "ppo_ac_positive_reward-shift-800000--adaptation-1000000--positive_likelihood-0-",
    ],
    help="the hyper parameters tuned",
)
parser.add_argument("--samples", type=int, default=500, help="the sampled point of the run")
parser.add_argument("--smooth-weight", type=float, default=0.90, help="the weight parameter of the exponential moving average")
parser.add_argument(
    "--last-n-episodes",
    type=int,
    default=50,
    help="for analysis only; the last n episodes from which the mean of the feature of interest is calculated",
)
parser.add_argument("--num-points-x-axis", type=int, default=500, help="the number of points in the x-axis")
parser.add_argument("--font-size", type=int, default=13, help="the font size of the plots")
parser.add_argument("--x-label", type=str, default="Time Steps", help="the label of x-axis")
parser.add_argument("--y-label", type=str, default="Average Episode Reward", help="the label of y-axis")
parser.add_argument("--y-lim-bottom", type=float, default=0.0, help="the bottom limit for the y-axis")
parser.add_argument("--output-format", type=str, default="pdf", help="either `pdf`, `png`, or `svg`")
parser.add_argument("--seed", type=int, default=6, help="seed of color palette shuffle")
args = parser.parse_args()
api = wandb.Api()
np.random.seed(args.seed)

# hacks
env_dict = {
    "MicrortsAttackShapedReward-v1": "MicrortsAttackHRL-v1",
    "MicrortsProduceCombatUnitsShapedReward-v1": "MicrortsProduceCombatUnitHRL-v1",
    "MicrortsRandomEnemyShapedReward3-v1": "MicrortsRandomEnemyHRL3-v1",
}
exp_convert_dict = {
    "ppo_positive_reward-positive_likelihood-0-": "sparse reward - no PLO",
    "ppo": "sparse reward",
    "ppo_ac_positive_reward-shift-2000000--adaptation-2000000--positive_likelihood-0-": "action guidance - multi-agent  w/ PLO",
    "ppo_ac_positive_reward-shift-2000000--adaptation-7000000--positive_likelihood-0-": "action guidance - long adaptation w/ PLO",
    "ppo_ac_positive_reward-shift-800000--adaptation-1000000--positive_likelihood-0-": "action guidance - short adaptation w/ PLO",
    "ppo_ac_positive_reward-shift-2000000--adaptation-7000000--positive_likelihood-1-": "action guidance - long adaptation",
    "ppo_ac_positive_reward-shift-800000--adaptation-1000000--positive_likelihood-1-": "action guidance - short adaptation",
    "pposhaped": "shaped reward",
}

# args.feature_of_interest = 'charts/episodic_return'
feature_name = args.feature_of_interest.replace("/", "_")
if not os.path.exists(feature_name):
    os.makedirs(feature_name)

if not path.exists(f"{feature_name}/all_df_cache.pkl"):
    # Change oreilly-class/cifar to <entity/project-name>
    runs = api.runs(args.wandb_project)
    summary_list = []
    config_list = []
    name_list = []
    envs = {}
    data = []
    exp_names = []

    for idx, run in enumerate(runs):
        if args.feature_of_interest in run.summary:
            # if args.scan_history:
            #     ls =
            # else:
            ls = run.history(keys=[args.feature_of_interest, "global_step"], pandas=False, samples=args.samples)
            metrics_dataframe = pd.DataFrame(ls[0])
            exp_name = run.config["exp_name"]
            for param in args.hyper_params_tuned:
                if param in run.config:
                    exp_name += "-" + param + "-" + str(run.config[param]) + "-"

            # hacks
            if run.config["env_id"] in env_dict:
                exp_name += "shaped"
                run.config["env_id"] = env_dict[run.config["env_id"]]

            metrics_dataframe.insert(len(metrics_dataframe.columns), "algo", exp_name)
            exp_names += [exp_name]
            metrics_dataframe.insert(len(metrics_dataframe.columns), "seed", run.config["seed"])

            data += [metrics_dataframe]
            if run.config["env_id"] not in envs:
                envs[run.config["env_id"]] = [metrics_dataframe]
                envs[run.config["env_id"] + "total_timesteps"] = run.config["total_timesteps"]
            else:
                envs[run.config["env_id"]] += [metrics_dataframe]

            # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)

            # run.config is the input metrics.  We remove special values that start with _.
            config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

            # run.name is the name of the run.
            name_list.append(run.name)

    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    name_df = pd.DataFrame({"name": name_list})
    all_df = pd.concat([name_df, config_df, summary_df], axis=1)
    data = pd.concat(data, ignore_index=True)

    with open(f"{feature_name}/all_df_cache.pkl", "wb") as handle:
        pickle.dump(all_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{feature_name}/envs_cache.pkl", "wb") as handle:
        pickle.dump(envs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f"{feature_name}/exp_names_cache.pkl", "wb") as handle:
        pickle.dump(exp_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f"{feature_name}/all_df_cache.pkl", "rb") as handle:
        all_df = pickle.load(handle)
    with open(f"{feature_name}/envs_cache.pkl", "rb") as handle:
        envs = pickle.load(handle)
    with open(f"{feature_name}/exp_names_cache.pkl", "rb") as handle:
        exp_names = pickle.load(handle)
print("data loaded")

# https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar#_=_
def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


# smoothing
for env in envs:
    if not env.endswith("total_timesteps"):
        for idx, metrics_dataframe in enumerate(envs[env]):
            envs[env][idx] = metrics_dataframe.dropna(subset=[args.feature_of_interest])
#             envs[env][idx][args.feature_of_interest] = smooth(metrics_dataframe[args.feature_of_interest], 0.85)

sns.set(style="darkgrid")


def get_df_for_env(env_id):
    env_total_timesteps = envs[env_id + "total_timesteps"]
    env_increment = env_total_timesteps / 500
    envs_same_x_axis = []
    for sampled_run in envs[env_id]:
        df = pd.DataFrame(columns=sampled_run.columns)
        x_axis = [i * env_increment for i in range(500 - 2)]
        current_row = 0
        for timestep in x_axis:
            while sampled_run.iloc[current_row]["global_step"] < timestep:
                current_row += 1
                if current_row > len(sampled_run) - 2:
                    break
            if current_row > len(sampled_run) - 2:
                break
            temp_row = sampled_run.iloc[current_row].copy()
            temp_row["global_step"] = timestep
            df = df.append(temp_row)

        envs_same_x_axis += [df]
    return pd.concat(envs_same_x_axis, ignore_index=True)


def export_legend(ax, filename="legend.pdf"):
    # import matplotlib as mpl
    # mpl.rcParams['text.usetex'] = True
    # mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis("off")
    handles, labels = ax.get_legend_handles_labels()

    legend = ax2.legend(
        handles=handles[1:], labels=labels[1:], frameon=False, loc="lower center", ncol=3, fontsize=20, handlelength=1
    )
    for text in legend.get_texts():
        text.set_text(exp_convert_dict[text.get_text()])
    for line in legend.get_lines():
        line.set_linewidth(4.0)
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    fig.clf()


if not os.path.exists(f"{feature_name}/data"):
    os.makedirs(f"{feature_name}/data")
if not os.path.exists(f"{feature_name}/plots"):
    os.makedirs(f"{feature_name}/plots")
if not os.path.exists(f"{feature_name}/legends"):
    os.makedirs(f"{feature_name}/legends")


interested_exp_names = sorted(list(set(exp_names)))  # ['ppo_continuous_action', 'ppo_atari_visual']
current_palette = sns.color_palette(n_colors=len(interested_exp_names))
np.random.shuffle(current_palette)
current_palette_dict = dict(zip(interested_exp_names, current_palette))
if args.interested_exp_names:
    interested_exp_names = args.interested_exp_names
print(current_palette_dict)
legend_df = pd.DataFrame()

if args.font_size:
    plt.rc("axes", titlesize=args.font_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=args.font_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=args.font_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=args.font_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=args.font_size)  # legend fontsize

stats = {item: [] for item in ["env_id", "exp_name", args.feature_of_interest]}
# uncommenet the following to generate all figures
for env in set(all_df["env_id"]):
    if not path.exists(f"{feature_name}/data/{env}.pkl"):
        with open(f"{feature_name}/data/{env}.pkl", "wb") as handle:
            data = get_df_for_env(env)
            data["seed"] = data["seed"].astype(float)
            data[args.feature_of_interest] = data[args.feature_of_interest].astype(float)
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f"{feature_name}/data/{env}.pkl", "rb") as handle:
            data = pickle.load(handle)
            print(f"{env}'s data loaded")

    def _smooth(df):
        df[args.feature_of_interest] = smooth(list(df[args.feature_of_interest]), args.smooth_weight)
        return df

    legend_df = legend_df.append(data)
    ax = sns.lineplot(
        data=data.groupby(["seed", "algo"]).apply(_smooth).loc[data["algo"].isin(interested_exp_names)],
        x="global_step",
        y=args.feature_of_interest,
        hue="algo",
        units="seed",
        estimator=None,
        palette=current_palette_dict,
        alpha=0.2,
    )
    sns.lineplot(
        data=data.groupby(["seed", "algo"]).apply(_smooth).loc[data["algo"].isin(interested_exp_names)],
        x="global_step",
        y=args.feature_of_interest,
        hue="algo",
        ci=None,
        palette=current_palette_dict,
        linewidth=2.0,
    )
    ax.set(xlabel=args.x_label, ylabel=args.y_label)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles=handles[1 : len(labels) // 2],
        labels=labels[1 : len(labels) // 2],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        fancybox=True,
    )
    for text in legend.get_texts():
        text.set_text(exp_convert_dict[text.get_text()])
    if args.y_lim_bottom:
        plt.ylim(bottom=args.y_lim_bottom)
    # plt.title(env)
    plt.tight_layout()
    plt.savefig(f"{feature_name}/plots/{env}.{args.output_format}")
    plt.clf()

    for algo in interested_exp_names:
        algo_data = data.loc[data["algo"].isin([algo])]
        last_n_episodes_global_step = sorted(algo_data["global_step"].unique())[-args.last_n_episodes]
        last_n_episodes_features = (
            algo_data[algo_data["global_step"] > last_n_episodes_global_step]
            .groupby(["seed"])
            .mean()[args.feature_of_interest]
        )

        for item in last_n_episodes_features:
            stats[args.feature_of_interest] += [item]
            stats["exp_name"] += [exp_convert_dict[algo]]
            stats["env_id"] += [env]

# export legend
ax = sns.lineplot(
    data=legend_df,
    x="global_step",
    y=args.feature_of_interest,
    hue="algo",
    ci="sd",
    palette=current_palette_dict,
)
ax.set(xlabel="Time Steps", ylabel="Average Episode Reward")
ax.legend().remove()
export_legend(ax, f"{feature_name}/legend.{args.output_format}")
plt.clf()


# analysis
stats_df = pd.DataFrame(stats)
g = stats_df.groupby(["env_id", "exp_name"]).agg(lambda x: f"{np.mean(x):.2f} ± {np.std(x):.2f}")
print(g.reset_index().pivot("exp_name", "env_id", args.feature_of_interest).to_latex().replace("±", "$\pm$"))
