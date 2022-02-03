# pip install boto3
import argparse
import re
import time
from distutils.util import strtobool

import boto3
import requests
import wandb

client = boto3.client("batch")

parser = argparse.ArgumentParser(description="CleanRL Experiment Submission")
# Common arguments
parser.add_argument(
    "--wandb-project", type=str, default="vwxyzjn/gym-microrts", help="the name of wandb project (e.g. cleanrl/cleanrl)"
)
parser.add_argument("--run-state", type=str, default="crashed", help="the name of this experiment")

parser.add_argument("--job-queue", type=str, default="cleanrl", help="the name of the job queue")
parser.add_argument(
    "--wandb-key", type=str, default="", help="the wandb key. If not provided, the script will try to read from `~/.netrc`"
)
parser.add_argument("--docker-repo", type=str, default="vwxyzjn/gym-microrts:latest", help="the name of the job queue")
parser.add_argument("--job-definition", type=str, default="gym-microrts", help="the name of the job definition")
parser.add_argument("--num-seed", type=int, default=2, help="number of random seeds for experiments")
parser.add_argument("--num-vcpu", type=int, default=1, help="number of vcpu per experiment")
parser.add_argument("--num-memory", type=int, default=2000, help="number of memory (MB) per experiment")
parser.add_argument("--num-gpu", type=int, default=0, help="number of gpu per experiment")
parser.add_argument("--num-hours", type=float, default=16.0, help="number of hours allocated experiment")
parser.add_argument(
    "--upload-files-baseurl", type=str, default="", help="the baseurl of your website if you decide to upload files"
)
parser.add_argument(
    "--submit-aws",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, script will need to be uploaded",
)
args = parser.parse_args()

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs(args.wandb_project)
run_ids = []
final_run_cmds = []
for run in runs:
    if run.state == args.run_state:
        run_ids += [run.path[-1]]
        metadata = requests.get(url=run.file(name="wandb-metadata.json").url).json()
        final_run_cmds += [["python", metadata["program"]] + metadata["args"]]
        if args.upload_files_baseurl:
            file_name = final_run_cmds[-1][1]
            link = args.upload_files_baseurl + "/" + file_name
            final_run_cmds[-1] = ["curl", "-O", link, ";"] + final_run_cmds[-1]

if not args.wandb_key:
    args.wandb_key = requests.utils.get_netrc_auth("https://api.wandb.ai")[-1]
assert (
    len(args.wandb_key) > 0
), "set the environment variable `WANDB_KEY` to your WANDB API key, something like `export WANDB_KEY=fdsfdsfdsfads` "

# use docker directly
if not args.submit_aws:
    cores = 40
    current_core = 0
    for run_id, final_run_cmd in zip(run_ids, final_run_cmds):
        print(
            f'docker run -d --cpuset-cpus="{current_core}" -e WANDB={wandb_key} -e WANDB_RESUME=must -e WANDB_RUN_ID={run_id} {args.docker_repo} '
            + '/bin/bash -c "'
            + " ".join(final_run_cmd)
            + '"'
        )
        current_core = (current_core + 1) % cores

# submit jobs
if args.submit_aws:
    for run_id, final_run_cmd in zip(run_ids, final_run_cmds):
        job_name = re.findall("(python)(.+)(.py)", " ".join(final_run_cmd))[0][1].strip() + str(int(time.time()))
        job_name = job_name.replace("/", "_").replace("_param ", "")
        resources_requirements = []
        if args.num_gpu:
            resources_requirements = [
                {"value": str(args.num_gpu), "type": "GPU"},
            ]

        response = client.submit_job(
            jobName=job_name,
            jobQueue=args.job_queue,
            jobDefinition=args.job_definition,
            containerOverrides={
                "vcpus": args.num_vcpu,
                "memory": args.num_memory,
                "command": ["/bin/bash", "-c", " ".join(final_run_cmd)],
                "environment": [
                    {"name": "WANDB", "value": wandb_key},
                    {"name": "WANDB_RESUME", "value": "must"},
                    {"name": "WANDB_RUN_ID", "value": run_id},
                ],
                "resourceRequirements": resources_requirements,
            },
            retryStrategy={"attempts": 1},
            timeout={"attemptDurationSeconds": int(args.num_hours * 60 * 60)},
        )
        if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            print(response)
            raise Exception("jobs submit failure")
