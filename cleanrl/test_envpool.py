# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""EnvPool python benchmark script.

Single Python Thread
====================

In general, we recommend to first decide batch_size, then with
num_envs = (2.5~3) * batch_size to achieve the best simulation speed.

In DGX-A100, we use batch_size 248 and num_envs 645.
In TPU-VM, we use batch_size 96 and num_envs 288.

NUMA
====

To test with NUMA, first use ``numactl -s`` to see the number of NUMA core;
then use ``./numa_test.sh `YOUR_CORE_NUM` python3 test_envpool.py [args]``.

In DGX-A100 (8 NUMA core), we use the following script to achieve overall
1M+ FPS:
::

  ./numa_test.sh 8 python3 test_envpool.py --num-envs 100 --batch-size 32 \
    --thread-affinity-offset -1

Note: When using NUMA, it's better to disable thread affinity by setting
`--thread-affinity-offset -1`.
"""

import argparse
import time

import envpool
import numpy as np
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="atari", choices=["atari", "mujoco", "vizdoom"])
    parser.add_argument("--num-envs", type=int, default=645)
    parser.add_argument("--batch-size", type=int, default=248)
    # num_threads == 0 means to let envpool itself determine
    parser.add_argument("--num-threads", type=int, default=0)
    # thread_affinity_offset == -1 means no thread affinity
    parser.add_argument("--thread-affinity-offset", type=int, default=0)
    parser.add_argument("--total-step", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(args)
    task_id = {
        "atari": "Pong-v5",
        "mujoco": "Ant-v3",
        "vizdoom": "HealthGathering-v1",
    }[args.env]
    kwargs = dict(
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        thread_affinity_offset=args.thread_affinity_offset,
    )
    if args.env in ["atari", "vizdoom"]:
        kwargs.update(use_inter_area_resize=False)
    env = envpool.make_gym(task_id, **kwargs)
    env.async_reset()
    env.action_space.seed(args.seed)
    action = np.array([env.action_space.sample() for _ in range(args.batch_size)])
    t = time.time()
    for _ in tqdm.trange(args.total_step):
        info = env.recv()[-1]
        env.send(action, info["env_id"])
    duration = time.time() - t
    frame_skip = getattr(env.spec.config, "frame_skip", 1)
    fps = args.total_step * args.batch_size / duration * frame_skip
    print(f"Duration = {duration:.2f}s")
    print(f"EnvPool FPS = {fps:.2f}")
