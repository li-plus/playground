# RAY_DEDUP_LOGS=0 python3 load_balance.py

import time
from datetime import datetime

import ray

task_time = [8, 2, 2, 4, 4, 2, 1, 1]


@ray.remote
def sleep(n):
    print(f"{datetime.now()}: task {n} launched")
    time.sleep(n)
    print(f"{datetime.now()}: task {n} finished")


ray.init(num_cpus=2)
ray.get([sleep.remote(t) for t in task_time])
