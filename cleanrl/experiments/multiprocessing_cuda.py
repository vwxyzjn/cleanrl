import torch
from torch import multiprocessing as mp
from faster_fifo import Queue as FastQueue

def producer(data_q):
    while True:
        data = [1,2,3]
        data_q.put(data)
 
def learner(data_q):
    while True:
        data = data_q.get()
        print(data)


if __name__ == '__main__':
    ctx = mp.get_context("spawn")
    data_q = FastQueue(1)
    procs = [
        ctx.Process(target=producer, args=(data_q,)) for _ in range(2)
    ]
    procs.append(ctx.Process(target=learner, args=(data_q,)))
    for p in procs: p.start()
    for p in procs: p.join()