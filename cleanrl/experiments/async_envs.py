
import os
# import threading
os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
import multiprocessing as pythonMp
from torch import multiprocessing as mp
# mp.set_start_method('forkserver')
from multiprocessing.managers import SyncManager, SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from faster_fifo import Queue as MpQueue
from queue import Empty
import stopwatch


class SharedNDArray(np.ndarray):
    def set_shm(self, shm):
        self.shm = shm

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()


def share_memory(arr):
    shm = SharedMemory(create=True, size=arr.nbytes)
    shm_arr = SharedNDArray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]
    shm_arr.set_shm(shm)
    return shm_arr


class AsyncEnvs:

    def __init__(self, env_fns, num_rollout_workers, num_steps, device, agent,
                 storage):
        # self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.num_rollout_workers = num_rollout_workers
        self.num_steps = num_steps
        self.device = device
        self.agent = agent
        ctx = mp.get_context("forkserver")
        self.rollout_task_queues = [ctx.Queue(10) for i in range(num_rollout_workers)]
        self.policy_request_queue = ctx.Queue(10)
        self.storage = storage
        
        assert len(env_fns) % self.num_rollout_workers == 0, \
            "number of rollout workers must divide the number of envs"
        self.num_envs_per_rollout_worker = len(env_fns) // self.num_rollout_workers
        
        for rollout_worker_idx in range(self.num_rollout_workers):
            ctx.Process(target=self.start_rollout_worker, args=(rollout_worker_idx,)).start()
        ctx.Process(target=self.start_policy_worker).start()

    def start_rollout_worker(self, rollout_worker_idx):
        sw = stopwatch.StopWatch()
        next_obs, next_done, obs, actions, logprobs, rewards, dones, values = self.storage
        rollout_task_queue = self.rollout_task_queues[rollout_worker_idx]
        env_idxs = range(rollout_worker_idx*self.num_envs_per_rollout_worker, 
                         rollout_worker_idx*self.num_envs_per_rollout_worker+self.num_envs_per_rollout_worker)
        for env_idx in env_idxs:
            next_step = 0
            self.policy_request_queue.put([next_step, env_idx, rollout_worker_idx])
            next_obs[env_idx] = torch.tensor(self.envs[env_idx].reset())
            next_done[env_idx] = 0
            print(env_idx)
            
        last_report = last_report_frames = total_env_frames = 0

        while True:
            with sw.timer('act'):
                with sw.timer('wait_rollout_task_queue'):
                    
                    tasks = []
                    for _ in range(4):
                        tasks.extend(self.rollout_task_queues[rollout_worker_idx].get())
                for task in tasks:
                    step, env_idx = task
                    with sw.timer('rollouts'):
                        obs[step,env_idx] = next_obs[env_idx].copy()
                        dones[step,env_idx] = next_done[env_idx].copy()
                        
                        next_obs[env_idx], r, d, info = self.envs[env_idx].step(actions[step,env_idx])
                        if d:
                            next_obs[env_idx] = self.envs[env_idx].reset()
                        rewards[step,env_idx] = r
                        next_done[env_idx] = d
                        next_step = (step + 1) % self.num_steps
                        self.policy_request_queue.put([next_step, env_idx, rollout_worker_idx])
                        if 'episode' in info.keys():
                            print(["charts/episode_reward", info['episode']['r']])
                            # stats_queue.put(['l', info['episode']['l']])
                            # stats_queue.put()

    def start_policy_worker(self):
        next_obs, next_done, obs, actions, logprobs, rewards, dones, values = self.storage
        sw = stopwatch.StopWatch()
        # min_num_requests = 3
        # wait_for_min_requests = 0.01
        # time.sleep(5)
        step = 0
        while True:
            step += 1
            with sw.timer('policy_worker'):
                # waiting_started = time.time()
                with sw.timer('policy_requests'):
                    # policy_requests = []
                    # while len(policy_requests) < min_num_requests and time.time() - waiting_started < wait_for_min_requests:
                    #     try:
                    #         policy_requests.extend(self.policy_request_queue.get_many(timeout=0.005))
                    #     except Empty:
                    #         pass
                    # if len(policy_requests) == 0:
                    #     continue
                    policy_requests = []
                    for _ in range(4):
                        policy_requests.extend(self.policy_request_queue.get())
                with sw.timer('prepare_data'):
                    ls = np.array(policy_requests)
                with sw.timer('index'):
                    next_o = next_obs[ls[:,1]]
                with sw.timer('inference'):
                    with torch.no_grad():
                        a, l, _ = self.agent.get_action(next_o)
                        v = self.agent.get_value(next_o)
                        print(a)
               
                for idx, item in enumerate(ls):
                    with sw.timer('move_to_cpu'):
                        actions[tuple(item[0,1])] = a[idx]
                        logprobs[tuple(item[0,1])] = l[idx]
                        values[tuple(item[0,1])] = v[idx]
                    with sw.timer('execute_action'):
                        self.rollout_task_queues[item[2]].put([item[0], item[1]])
                    # for idx, item in enumerate(ls[:,[3]]):
                    #     actions[tuple(item)] = a[idx]
                    #     logprobs[tuple(item)] = l[idx]
                    #     values[tuple(item)] = v[idx]
                    # for rollout_worker_idx in range(self.num_rollout_workers):
                    #     rollout_worker_idx = rollout_worker_idxs[j]
                    #     split_idx = split_idxs[j]
                    #     step_idx = step_idxs[j]
                    #     rollout_task_queues[rollout_worker_idx].put([split_idx,step_idx])