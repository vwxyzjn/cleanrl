# pip install boto3
import boto3
import re
import time
import os
client = boto3.client('batch')

# get env variable values
wandb_key = os.environ['WANDB_KEY']
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
for final_run_cmd in final_run_cmds:
    job_name = re.findall('(python)(.+)(.py)'," ".join(final_run_cmd))[0][1].strip() + str(int(time.time()))
    job_name = job_name.replace("/", "_")
    response = client.submit_job(
        jobName=job_name,
        jobQueue='cleanrl',
        jobDefinition='cleanrl',
        containerOverrides={
            'vcpus': 1,
            'memory': 2000,
            'command': final_run_cmd,
            'environment': [
                {
                    'name': 'WANDB',
                    'value': wandb_key
                },
                {
                    'name': 'MKL_NUM_THREADS',
                    'value': "1"
                },
                {
                    'name': 'OMP_NUM_THREADS',
                    'value': "1"
                },
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