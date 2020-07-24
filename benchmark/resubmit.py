# pip install boto3
import boto3
import re
import time
import os
import requests
import json
import argparse
import wandb
import requests
from distutils.util import strtobool
client = boto3.client('batch')

parser = argparse.ArgumentParser(description='CleanRL Experiment Submission')
# Common arguments
parser.add_argument('--wandb-project', type=str, default="cleanrl/cleanrl.benchmark",
                   help='the name of wandb project (e.g. cleanrl/cleanrl)')
parser.add_argument('--run-state', type=str, default="crashed",
                   help='the name of this experiment')
parser.add_argument('--job-queue', type=str, default="cleanrl",
                   help='the name of the job queue')
parser.add_argument('--job-definition', type=str, default="cleanrl",
                   help='the name of the job definition')
parser.add_argument('--num-vcpu', type=int, default=2,
                   help='number of vcpu per experiment')
parser.add_argument('--num-memory', type=int, default=15000,
                   help='number of memory (MB) per experiment')
parser.add_argument('--num-gpu', type=int, default=1,
                   help='number of gpu per experiment')
parser.add_argument('--num-hours', type=float, default=48.0,
                   help='number of hours allocated experiment')
parser.add_argument('--upload_files', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                    help='if toggled, script will need to be uploaded')
parser.add_argument('--submit-aws', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                    help='if toggled, script will need to be uploaded')
args = parser.parse_args()

api = wandb.Api()

if args.upload_files:
    response = requests.get('http://127.0.0.1:4040/api/tunnels')
    content = json.loads(response.content.decode())
    assert response.status_code == 200
    url = content['tunnels'][0]['public_url']

# Project is specified by <entity/project-name>
runs = api.runs(args.wandb_project)
final_run_cmds = []
for run in runs:
    if run.state == args.run_state:
        metadata = requests.get(url=run.file(name="wandb-metadata.json").url).json()
        final_run_cmds += [["python", metadata["program"]] + metadata["args"]]
        if args.upload_files:
            file_name = final_run_cmds[-1][1]
            link = url + '/' + file_name
            final_run_cmds[-1] = ['wget', link, ';'] + final_run_cmds[-1]

# get env variable values
wandb_key = os.environ['WANDB_KEY']
assert len(wandb_key) > 0, "set the environment variable `WANDB_KEY` to your WANDB API key, something like `export WANDB_KEY=fdsfdsfdsfads` "

# use docker directly
if not args.submit_aws:
    cores = 40
    repo = "vwxyzjn/cleanrl:latest"
    current_core = 0
    for final_run_cmd in final_run_cmds:
        print(f'docker run -d --cpuset-cpus="{current_core}" -e WANDB={wandb_key} {repo} ' + 
            '/bin/bash -c "' + " ".join(final_run_cmd) + '"')
        current_core = (current_core + 1) % cores

# submit jobs
if args.submit_aws:
    for final_run_cmd in final_run_cmds:
        job_name = re.findall('(python)(.+)(.py)'," ".join(final_run_cmd))[0][1].strip() + str(int(time.time()))
        job_name = job_name.replace("/", "_").replace("_param ", "")
        resources_requirements = []
        if args.num_gpu:
            resources_requirements = [
                {
                    'value': '1',
                    'type': 'GPU'
                },
            ]
        
        response = client.submit_job(
            jobName=job_name,
            jobQueue=args.job_queue,
            jobDefinition=args.job_definition,
            containerOverrides={
                'vcpus': args.num_vcpu,
                'memory': args.num_memory,
                'command': ["/bin/bash", "-c", " ".join(final_run_cmd)],
                'environment': [
                    {
                        'name': 'WANDB',
                        'value': wandb_key
                    }
                ],
                'resourceRequirements': resources_requirements,
            },
            retryStrategy={
                'attempts': 1
            },
            timeout={
                'attemptDurationSeconds': int(args.num_hours*60*60)
            }
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            print(response)
            raise Exception("jobs submit failure")

