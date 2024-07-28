# RAY_DEDUP_LOGS=0 python3 load_balance.py

import os
import ray
import time
from datetime import datetime

task_time = [8, 2, 2, 4, 4, 2, 1, 1]

@ray.remote(num_cpus=os.cpu_count() // 2)
def sleep(n):
    print(f"{datetime.now()}: task {n} launched")
    time.sleep(n)
    print(f"{datetime.now()}: task {n} finished")

ray.get([sleep.remote(t) for t in task_time])
