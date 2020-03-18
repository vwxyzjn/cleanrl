# Or run the experiments directly by 
# nomad job dispatch -meta "wandb=$WANDB_KEY" cleanrl

import nomad
import re

# get env variable values
wandb_key = os.environ['WANDB_KEY']
nomad_server_address = ""
assert len(wandb_key) > 0, "set the environment variable `WANDB_KEY` to your WANDB API key, something like `export WANDB_KEY=fdsfdsfdsfads` "

# extract runs from bash scripts
final_run_cmds = []
with open("test.sh") as f:
    strings = f.read()
runs_match = re.findall('(python)(.+)((?:\n.+)+)(seed)',strings)
for run_match in runs_match:
    run_match_str = "".join(run_match).replace("\\\n", "")
    # print(run_match_str)
    for seed in range(2):
        final_run_cmds += [run_match_str.replace("$seed", str(seed)).split()]

# submit jobs
n = nomad.Nomad(
    host=nomad_server_address,
)
for final_run_cmd in final_run_cmds:
    n.job.dispatch_job("cleanrl_job", meta={
        "wandb": wandb_key,
        "command": " ".join(final_run_cmd)
    })
