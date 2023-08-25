import torch
import time

a = torch.ones(1024, 1024, 1024).pin_memory()

start = time.time()
a.cuda()
print(f'blocking_cpu_elapsed {time.time() - start:.4f}')
torch.cuda.synchronize()
print(f'blocking_sync_elapsed {time.time() - start:.4f}')

start = time.time()
a.cuda(non_blocking=True)
print(f'non_blocking_cpu_elapsed {time.time() - start:.4f}s')
torch.cuda.synchronize()
print(f'non_blocking_sync_elapsed {time.time() - start:.4f}s')
