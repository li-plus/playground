import os
import random

import ray

ray.init()


@ray.remote(num_cpus=os.cpu_count(), max_retries=0)
def do_work(index):
    if random.random() < 0.5:
        print("this work is too hard for me, exiting ...")
        exit(1)
    return index


obj_refs = [do_work.remote(i) for i in range(10)]

results = []
for obj_ref in obj_refs:
    try:
        ret = ray.get(obj_ref)
    except Exception:
        ret = -1
    results.append(ret)

print("final results", results)
