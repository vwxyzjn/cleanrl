# This script look for early-terminated runs in a wanbd project and resubmit through aws

import wandb
import requests
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("cleanrl/ppo.kle.noent")
final_run_cmds = []
for run in runs:
    if run.state == "failed":
        metadata = requests.get(url=run.file(name="wandb-metadata.json").url).json()
        final_run_cmds += [["python", metadata["program"]] + metadata["args"]]


# pip install boto3
import boto3
import re
import time
import os
client = boto3.client('batch')

# get env variable values
wandb_key = os.environ['WANDB_KEY']
assert len(wandb_key) > 0, "set the environment variable `WANDB_KEY` to your WANDB API key, something like `export WANDB_KEY=fdsfdsfdsfads` "

# use docker directly
cores = 40
repo = "vwxyzjn/cleanrl_shared_memory:latest"
current_core = 1
for final_run_cmd in final_run_cmds:
    print(f'docker run -d --cpuset-cpus="{current_core}" -e WANDB={wandb_key} {repo} ' + " ".join(final_run_cmd))
    print("\n")
    current_core = current_core + 1 % cores

# submit jobs
for final_run_cmd in final_run_cmds:
    job_name = re.findall('(python)(.+)(.py)'," ".join(final_run_cmd))[0][1].strip() + str(int(time.time()))
    job_name = job_name.replace("/", "_")
    response = client.submit_job(
        jobName=job_name,
        jobQueue='cleanrl',
        jobDefinition='cleanrl',
        containerOverrides={
            'vcpus': 1,
            'memory': 300,
            'command': final_run_cmd,
            'environment': [
                {
                    'name': 'WANDB',
                    'value': wandb_key
                }
            ]
        },
        retryStrategy={
            'attempts': 1
        },
        timeout={
            'attemptDurationSeconds': 16*60*60 # 16 hours
        }
    )
    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print(response)
        raise Exception("jobs submit failure")