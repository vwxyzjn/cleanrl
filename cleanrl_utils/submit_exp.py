# pip install boto3
import boto3
import re
import time
import os
import requests
import json
import wandb
import argparse
import subprocess
import multiprocessing
from distutils.util import strtobool
client = boto3.client('batch')

parser = argparse.ArgumentParser(description='CleanRL Experiment Submission')
# experiment generation
parser.add_argument('--exp-script', type=str, default="debug.sh",
                    help='the file name of this experiment')
parser.add_argument('--algo', type=str, default="ppo.py",
                    help='the algorithm that will be used')
parser.add_argument('--gym-ids', nargs='+', 
                    help='the ids of the gym environment')
parser.add_argument('--total-timesteps', type=int, default=int(1e9),
                    help='total timesteps of the experiments')
parser.add_argument('--other-args', type=str, default="",
                    help="the entity (team) of wandb's project")

# experiment submission
parser.add_argument('--job-queue', type=str, default="cleanrl",
                   help='the name of the job queue')
parser.add_argument('--wandb-key', type=str, default="",
                   help='the wandb key. If not provided, the script will try to read from `~/.netrc`')
parser.add_argument('--docker-repo', type=str, default="vwxyzjn/cleanrl:latest",
                   help='the name of the job queue')
parser.add_argument('--job-definition', type=str, default="cleanrl",
                   help='the name of the job definition')
parser.add_argument('--num-seed', type=int, default=2,
                   help='number of random seeds for experiments')
parser.add_argument('--num-vcpu', type=int, default=1,
                   help='number of vcpu per experiment')
parser.add_argument('--num-memory', type=int, default=2000,
                   help='number of memory (MB) per experiment')
parser.add_argument('--num-gpu', type=int, default=0,
                   help='number of gpu per experiment')
parser.add_argument('--num-hours', type=float, default=16.0,
                   help='number of hours allocated experiment')
parser.add_argument('--auto-resume-attempts', type=int, default=1,
                   help='the name of the job queue')
parser.add_argument('--upload-files-baseurl', type=str, default="",
                   help='the baseurl of your website if you decide to upload files')
parser.add_argument('--submit-aws', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                    help='if toggled, script will need to be uploaded')

args = parser.parse_args()

subprocess.run(
    f"docker build -t {args.docker_repo} .",
    shell=True,
    check=True,
)

if not args.wandb_key:
    args.wandb_key = requests.utils.get_netrc_auth("https://api.wandb.ai")[-1]
assert len(args.wandb_key) > 0, "you have not logged into W&B; try do `wandb login`"

# extract runs from bash scripts
final_run_cmds = []
for seed in range(1,1+args.num_seed):
    final_run_cmds += [["python", args.algo, args.other_args, "--seed", str(seed)]]
    if args.upload_files_baseurl:
        file_name = final_run_cmds[-1][1]
        link = args.upload_files_baseurl + '/' + file_name
        final_run_cmds[-1] = ['curl', '-O', link, ';'] + final_run_cmds[-1]


run_ids = [wandb.util.generate_id() for _ in range (args.num_seed)]

final_str = ""
cores = multiprocessing.cpu_count()
current_core = 0
for run_id, final_run_cmd in zip(run_ids, final_run_cmds):
    run_command = (f'docker run -d --cpuset-cpus="{current_core}" -e WANDB={args.wandb_key} -e WANDB_RESUME=allow -e WANDB_RUN_ID={run_id} {args.docker_repo} ' + 
        '/bin/bash -c "' + " ".join(final_run_cmd) + '"' + "\n")
    print(run_command)
    final_str += run_command
    current_core = (current_core + 1) % cores

with open(f"{args.exp_script}.docker.sh", "w+") as f:
    f.write(final_str)

# submit jobs
if args.submit_aws:
    for run_id, final_run_cmd in zip(run_ids, final_run_cmds):
        job_name = re.findall('(python)(.+)(.py)'," ".join(final_run_cmd))[0][1].strip() + str(int(time.time()))
        job_name = job_name.replace("/", "_").replace("_param ", "")
        resources_requirements = []
        if args.num_gpu:
            resources_requirements = [
                {
                    'value': str(args.num_gpu),
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
                    {'name': 'WANDB', 'value': args.wandb_key},
                    {'name': 'WANDB_RESUME', 'value': 'allow'},
                    {'name': 'WANDB_RUN_ID', 'value': run_id},
                ],
                'resourceRequirements': resources_requirements,
            },
            retryStrategy={
                'attempts': args.auto_resume_attempts
            },
            timeout={
                'attemptDurationSeconds': int(args.num_hours*60*60)
            }
        )
        if response['ResponseMetadata']['HTTPStatusCode'] != 200:
            print(response)
            raise Exception("jobs submit failure")
