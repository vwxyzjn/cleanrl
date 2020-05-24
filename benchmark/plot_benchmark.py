from os import path
import pickle
import wandb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
api = wandb.Api()

if not path.exists("all_df_cache.pkl"):
    # Change oreilly-class/cifar to <entity/project-name>
    runs = api.runs("cleanrl/cleanrl.benchmark")
    summary_list = [] 
    config_list = [] 
    name_list = []
    envs = {}
    data = []
    rolling_average = 10
    sample_points = 500
    
    for idx, run in enumerate(runs):
        ls = run.history(keys=['charts/episode_reward', 'global_step'], pandas=False)
        metrics_dataframe = pd.DataFrame(ls[0])
        metrics_dataframe.insert(len(metrics_dataframe.columns), "algo", run.config['exp_name'])
        metrics_dataframe.insert(len(metrics_dataframe.columns), "seed", run.config['seed'])
        metrics_dataframe["charts/episode_reward"] = metrics_dataframe["charts/episode_reward"].rolling(rolling_average).mean()[rolling_average:]
        data += [metrics_dataframe]
        if run.config["gym_id"] not in envs:
            envs[run.config["gym_id"]] = [metrics_dataframe]
            envs[run.config["gym_id"]+"total_timesteps"] = run.config["total_timesteps"]
        else:
            envs[run.config["gym_id"]] += [metrics_dataframe]
    
        
        # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files 
        summary_list.append(run.summary._json_dict) 
    
        # run.config is the input metrics.  We remove special values that start with _.
        config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')}) 
    
        # run.name is the name of the run.
        name_list.append(run.name)       
    
    
    summary_df = pd.DataFrame.from_records(summary_list) 
    config_df = pd.DataFrame.from_records(config_list) 
    name_df = pd.DataFrame({'name': name_list}) 
    all_df = pd.concat([name_df, config_df,summary_df], axis=1)
    data = pd.concat(data, ignore_index=True)
    
    
    with open('all_df_cache.pkl', 'wb') as handle:
        pickle.dump(all_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('envs_cache.pkl', 'wb') as handle:
        pickle.dump(envs, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('all_df_cache.pkl', 'rb') as handle:
        all_df = pickle.load(handle)
    with open('envs_cache.pkl', 'rb') as handle:
        envs = pickle.load(handle)
        

sns.set(style="darkgrid")
def get_df_for_env(gym_id):
    env_total_timesteps = envs[gym_id+"total_timesteps"]
    env_increment = env_total_timesteps / 500
    envs_same_x_axis = []
    for sampled_run in envs[gym_id]:
        df = pd.DataFrame(columns=sampled_run.columns)
        x_axis = [i*env_increment for i in range(500-2)]
        current_row = 0
        for timestep in x_axis:
            while sampled_run.iloc[current_row]["global_step"] < timestep:
                current_row += 1
                if current_row > len(sampled_run)-2:
                    break
            if current_row > len(sampled_run)-2:
                break
            temp_row = sampled_run.iloc[current_row].copy()
            temp_row["global_step"] = timestep
            df = df.append(temp_row)
        
        envs_same_x_axis += [df]
    return pd.concat(envs_same_x_axis, ignore_index=True)

# uncommenet the following to generate all figures
# for env in set(all_df["gym_id"]):
#     data = get_df_for_env(env)
#     sns.lineplot(data=data, x="global_step", y="charts/episode_reward", hue="algo", ci='sd')
#     plt.legend(fontsize=6)
#     plt.title(env)
#     plt.savefig(f"{env}.svg")
#     plt.clf()

# debugging
env = "InvertedPendulumBulletEnv-v0"
data = get_df_for_env(env)
sns.lineplot(data=data, x="global_step", y="charts/episode_reward", hue="algo", ci='sd')
plt.legend(fontsize=6)
plt.title(env)