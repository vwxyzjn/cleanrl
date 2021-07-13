import torch
import nvidia_smi

from torch import multiprocessing as mp
# Receiver
def receiver(x):
    x[0] = 200
    # print(x)
 

if __name__ == '__main__':
    ctx = mp.get_context("fork")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.arange(10).reshape(5, 2)
    x.share_memory_()
    cudart = torch.cuda.cudart()
    r = cudart.cudaHostRegister(x.data_ptr(), x.numel() * x.element_size(), 0)
    assert x.is_shared()
    assert x.is_pinned()
    g = x.numpy()

    procs = [
        ctx.Process(target=receiver, args=(g,)) for _ in range(1)
    ]
    for p in procs: p.start()
        