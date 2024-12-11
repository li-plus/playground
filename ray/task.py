"""
Run locally: python3 hello_world.py

Submit to ray cluster using ray job api:
ray start --head
RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python3 hello_world.py

See https://docs.ray.io/en/latest/cluster/running-applications/job-submission/index.html
"""

import os
import time

import ray


@ray.remote
def add_remote(start, end):
    return sum(range(start, end))


def add_local(start, end):
    return sum(range(start, end))


ray.init()

N = 1000000000

start = time.time()
result = add_local(0, N)
elapsed = time.time() - start
print(f"add_local {result=} {elapsed=:.3f} s")

start = time.time()
num_cpus = os.cpu_count()
chunk_size = (N + num_cpus - 1) // num_cpus
result = sum(ray.get([add_remote.remote(i * chunk_size, min((i + 1) * chunk_size, N)) for i in range(num_cpus)]))
elapsed = time.time() - start
print(f"add_remote {result=} {elapsed=:.3f} s")
