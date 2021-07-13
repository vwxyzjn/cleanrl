# Reference: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
import torch
import numpy as np
import time

# import threading
# os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
from torch import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from faster_fifo import Queue as MpQueue

class SharedNDArray(np.ndarray):
    def set_shm(self, shm):
        self.shm = shm

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()

def to_numpy(t, num_dimensions):
    arr_shape = t.shape[:num_dimensions]
    arr = np.ndarray(arr_shape, dtype=object)
    to_numpy_func(t, arr)
    return arr

def to_numpy_func(t, arr):
    if len(arr.shape) == 1:
        for i in range(t.shape[0]):
            arr[i] = t[i]
    else:
        for i in range(t.shape[0]):
            to_numpy_func(t[i], arr[i])

def share_memory_torch_numpy_mixed(arr, num_dimensions):
    t = torch.tensor(arr)
    t.share_memory_()
    return to_numpy(t, 5)

def share_memory_numpy(arr):
    shm = SharedMemory(create=True, size=arr.nbytes)
    shm_arr = SharedNDArray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]
    shm_arr.set_shm(shm)
    return shm_arr


def act(obs, num_envs, num_steps):
    last_report = last_report_frames = total_env_frames = 0
    while True:
        for env_idx in range(num_envs):
            for step in range(num_steps):
                o = np.random.rand(84,84,3)
                # obs[i,env_idx,0,0,step] = torch.from_numpy(o)
                
                obs[i,env_idx,0,0,step] = o
                total_env_frames += 1
                    
            now = time.time()
            if now - last_report > 1:
                last_report = now
                frames_since_last_report = total_env_frames - last_report_frames
                last_report_frames = total_env_frames
                stats_queue.put(frames_since_last_report)
                # self.report_queue.put(dict(proc_idx=proc_idx, env_frames=frames_since_last_report))

if __name__ == "__main__":
    num_cpus = 4
    num_envs = 20
    num_steps = 32
    
    lock = mp.Lock()
    dimensions = (
        mp.cpu_count(),
        num_envs,
        1,
        1,
        num_steps,
    )
    # obs = share_memory_numpy(np.zeros(dimensions + (84,84,3)))
    obs = share_memory_torch_numpy_mixed(np.zeros(dimensions + (84,84,3)), 5)
    # raise
    actor_processes = []
    ctx = mp.get_context("forkserver")
    stats_queue = MpQueue()
    for i in range(num_cpus):
        actor = mp.Process(
            target=act,
            args=[obs, num_envs, num_steps],
        )
        actor.start()
        actor_processes.append(actor)
    import timeit
    timer = timeit.default_timer
    existing_video_files = []
    global_step = 0
    global_step_increment = 0
    start_time = time.time()
    update_step = 0
    try:
        while global_step < 100000000:
            update_step += 1
            try:
                ls = stats_queue.get_many(timeout=1)
                for l in ls:
                    global_step_increment += l
            except:
                continue
            if update_step % 5 == 0:
                print(f"global_step={global_step}")
                global_step += global_step_increment
                print("FPS: ", global_step_increment / (time.time() - start_time))
                global_step_increment = 0
                start_time = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        for actor in actor_processes:
            actor.terminate()
            actor.join(timeout=1)
