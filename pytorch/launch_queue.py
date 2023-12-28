import itertools
import time

import torch

a = torch.ones(10240, 10240, device="cuda")

for i in itertools.count():
    start = time.time()
    a @ a
    cost = time.time() - start
    print(f"submitted kernel {i}, elapsed {cost:.4f} seconds")

# launch queue will start blocking at kernel 1022, which is the queue size
