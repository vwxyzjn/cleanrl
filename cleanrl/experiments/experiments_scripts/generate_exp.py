template = '''
for seed in {{1..2}}
do
    (sleep 0.3 && nohup xvfb-run -a python ppo2_continuous_actions.py \\
    --gym-id PongNoFrameskip-v4 \\
    --total-timesteps 1000000 \\
    --wandb-project-name cleanrl \\
    --wandb-entity cleanrl \\
    --prod-mode \\
    {0} \\
    --capture-video \\
    --seed $seed
    ) >& /dev/null &
done
'''
features = ['--kl','--gae','--norm-obs','--norm-rewards','--norm-returns','--no-obs-reset','--no-reward-reset','--anneal-lr']
num_process = 3
count = 0

final_str = ""

# no feature toggled by default, each feature toggle at a time
for feature in features:
    final_str += template.format(feature)
    count+=1
    if count % num_process == 0:
        final_str += "\nwait\n"
# all features toggled by default, each feature turned off at a time
for feature in features:
    remains = list(set(features) - set([feature]))
    final_str += template.format(" ".join(remains))
    count+=1
    if count % num_process == 0:
        final_str += "\nwait\n"

# '--kl' toggled by default, each feature toggle at a time
for feature in list(set(features) - set(['--kl'])):
    final_str += template.format(" ".join(['--kl', feature]))
    count+=1
    if count % num_process == 0:
        final_str += "\nwait\n"