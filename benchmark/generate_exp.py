import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description='CleanRL Experiment Submission')
# Common arguments
parser.add_argument('--exp-script', type=str, default="exp.sh",
                    help='the file name of this experiment')
parser.add_argument('--algo', type=str, default="ppo.py",
                    help='the algorithm that will be used')
parser.add_argument('--gym-ids', nargs='+', 
                    help='the ids of the gym environment')
parser.add_argument('--total-timesteps', type=int, default=int(1e9),
                    help='total timesteps of the experiments')
parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                    help='if toggled, cuda will not be enabled by default')
parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                    help="the wandb's project name")
parser.add_argument('--wandb-entity', type=str, default=None,
                    help="the entity (team) of wandb's project")
args = parser.parse_args()

template = '''
for seed in {{1..2}}
do
    (sleep 0.3 && nohup xvfb-run -a python {} \\
    --gym-id {} \\
    --total-timesteps {} \\
    --wandb-project-name {} \\
    --wandb-entity {} \\
    --prod-mode \\
    {} \\
    --capture-video \\
    --seed $seed
    ) >& /dev/null &
done
'''

final_str = ""
for env in args.gym_ids:
    # all other features toggled by default
    if args.cuda:
        final_str += template.format(args.algo, env, args.total_timesteps, args.wandb_project_name, args.wandb_entity, "--cuda")
    else:
        final_str += template.format(args.algo, env, args.total_timesteps, args.wandb_project_name, args.wandb_entity, "--cuda False")

with open(f"{args.exp_script}", "w+") as f:
    f.write(final_str)