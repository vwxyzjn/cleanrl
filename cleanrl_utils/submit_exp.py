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

# fmt: off
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
    help="the other arguments of the script")

# experiment submission
parser.add_argument('--job-queue', type=str, default="",
    help='the name of the job queue')
parser.add_argument('--instance', type=str, default="a1.medium",
    help='the name of the job queue')
parser.add_argument('--wandb-key', type=str, default="",
    help='the wandb key. If not provided, the script will try to read from `~/.netrc`')
parser.add_argument('--docker-repo', type=str, default="vwxyzjn/cleanrl:latest",
    help='the name of the job queue')
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
parser.add_argument('--build-n-push', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
    help='if toggled, the script will build a container and push using `Dockerfile`')
parser.add_argument('--multi-archs', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
    help='if toggled, the script will build a container for both ARMs and AMD64')
parser.add_argument('--provider', type=str, default="", choices=["aws"],
    help='the cloud provider of choice (currently only `aws` is supported)')
args = parser.parse_args()
# fmt: on

if args.build_n_push:
    if args.multi_archs:
        subprocess.run(
            f"docker buildx build --push --platform linux/arm64,linux/amd64 -t {args.docker_repo} .",
            shell=True,
            check=True,
        )
    else:
        subprocess.run(
            f"docker build -t {args.docker_repo} .",
            shell=True,
            check=True,
        )
        subprocess.run(
            f"docker push {args.docker_repo}",
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

final_str = ""
cores = multiprocessing.cpu_count()
current_core = 0
for final_run_cmd in final_run_cmds:
    run_command = (f'docker run -d --cpuset-cpus="{current_core}" -e WANDB={args.wandb_key} {args.docker_repo} ' + 
        '/bin/bash -c "' + " ".join(final_run_cmd) + '"' + "\n")
    print(run_command)
    final_str += run_command
    current_core = (current_core + 1) % cores

with open(f"{args.exp_script}.docker.sh", "w+") as f:
    f.write(final_str)

# submit jobs
if args.provider == "aws":
    for final_run_cmd in final_run_cmds:
        job_name = args.algo.replace(".py", "") + str(int(time.time()))
        resources_requirements = []
        if args.num_gpu:
            resources_requirements = [
                {
                    'value': str(args.num_gpu),
                    'type': 'GPU'
                },
            ]
        try:
            job_def_name = args.docker_repo.replace(":", "_").replace("/", "_")
            job_def = client.register_job_definition(
                jobDefinitionName=job_def_name,
                type='container',
                containerProperties={
                    'image': args.docker_repo,
                    'vcpus': args.num_vcpu,
                    'memory': args.num_memory,
                    'command': [
                        '/bin/bash',
                    ],
                }
            )
            response = client.submit_job(
                jobName=job_name,
                jobQueue=args.job_queue,
                jobDefinition=job_def_name,
                containerOverrides={
                    'vcpus': args.num_vcpu,
                    'memory': args.num_memory,
                    'command': ["/bin/bash", "-c", " ".join(final_run_cmd)],
                    'environment': [
                        {'name': 'WANDB', 'value': args.wandb_key},
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
        except Exception as e:
            print(e)
        finally:
            response = client.deregister_job_definition(
                jobDefinition=job_def_name
            )
            if response['ResponseMetadata']['HTTPStatusCode'] != 200:
                print(response)
                raise Exception("jobs submit failure")
