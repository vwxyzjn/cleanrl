"""
See https://github.com/docker/docker-py/issues/2395
At the moment, nvidia-container-toolkit still includes nvidia-container-runtime. So, you can still add nvidia-container-runtime as a runtime in /etc/docker/daemon.json:

{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
Then restart the docker service (sudo systemctl restart docker) and use runtime="nvidia" in docker-py as before.
"""


import argparse
import shlex
import time

import docker

parser = argparse.ArgumentParser(description="CleanRL Docker Submission")
# Common arguments
parser.add_argument("--exp-script", type=str, default="test1.sh", help="the file name of this experiment")
# parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
#                     help='if toggled, cuda will not be enabled by default')
parser.add_argument("--num-vcpus", type=int, default=16, help="total number of vcpus used in the host machine")
parser.add_argument("--frequency", type=int, default=1, help="the number of seconds to check container update status")
args = parser.parse_args()

client = docker.from_env()

# c = client.containers.run("ubuntu:latest", "echo hello world", detach=True)

with open(args.exp_script) as f:
    lines = f.readlines()

tasks = []
for line in lines:
    line.replace("\n", "")
    line_split = shlex.split(line)
    for idx, item in enumerate(line_split):
        if item == "-e":
            break
    env_vars = line_split[idx + 1 : idx + 2]
    image = line_split[idx + 2]
    commands = line_split[idx + 3 :]
    tasks += [[image, env_vars, commands]]

running_containers = []
vcpus = list(range(args.num_vcpus))
while len(tasks) != 0:
    time.sleep(args.frequency)

    # update running_containers
    new_running_containers = []
    for item in running_containers:
        c = item[0]
        c.reload()
        if c.status != "exited":
            new_running_containers += [item]
        else:
            print(f"✅ task on vcpu {item[1]} has finished")
            vcpus += [item[1]]
    running_containers = new_running_containers

    if len(vcpus) != 0:
        task = tasks.pop()
        vcpu = vcpus.pop()
        # if args.cuda:
        #     c = client.containers.run(
        #         image=task[0],
        #         environment=task[1],
        #         command=task[2],
        #         runtime="nvidia",
        #         cpuset_cpus=str(vcpu),
        #         detach=True)
        #     running_containers += [[c, vcpu]]
        # else:
        c = client.containers.run(image=task[0], environment=task[1], command=task[2], cpuset_cpus=str(vcpu), detach=True)
        running_containers += [[c, vcpu]]
        print("========================")
        print(f"remaining tasks={len(tasks)}, running containers={len(running_containers)}")
        print(f"running on vcpu {vcpu}", task)
