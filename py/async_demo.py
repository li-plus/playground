"""
docs: https://docs.python.org/3/library/asyncio-task.html
tutorial: https://medium.com/@moraneus/mastering-pythons-asyncio-a-practical-guide-0a673265cf04
"""

import asyncio
import concurrent.futures
import logging
import os
import random
import threading
import time

os.environ.update({"RAY_DEDUP_LOGS": "0"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


async def hello_world():
    print("hello")
    await asyncio.sleep(1)
    print("world")


async def work_concurrent():
    async def worker(delay):
        await asyncio.sleep(delay)
        return delay

    start = time.time()
    rets = await asyncio.gather(worker(delay=2), worker(delay=1))
    print(f"Finished within {time.time() - start:.2f}s. Return values: {rets}")


async def coroutine_pool():
    num_workers = 10
    num_tasks = 100

    async def worker_fn(worker_id: int, queue: asyncio.Queue):
        while True:
            delay = await queue.get()
            try:
                await asyncio.sleep(delay)
                print(f"[worker {worker_id}] worked for {delay:.3f}s")
            finally:
                queue.task_done()

    queue = asyncio.Queue()
    for _ in range(num_tasks):
        await queue.put(random.random())

    workers = [asyncio.create_task(worker_fn(i, queue)) for i in range(num_workers)]

    # wait until all tasks are done
    await queue.join()

    # cancel all workers: this
    for worker in workers:
        worker.cancel()

    # wait for cancellation
    await asyncio.gather(*workers, return_exceptions=True)


async def coroutine_pool_with_task_group():
    num_workers = 10
    num_tasks = 100

    async def worker_fn(worker_id: int, queue: asyncio.Queue):
        while True:
            delay = await queue.get()
            try:
                if delay is None:
                    break
                await asyncio.sleep(delay)
                print(f"[worker {worker_id}] worked for {delay:.3f}s")
            finally:
                queue.task_done()

    queue = asyncio.Queue()
    for _ in range(num_tasks):
        await queue.put(random.random())
    for _ in range(num_workers):
        await queue.put(None)

    async with asyncio.TaskGroup() as tg:
        for i in range(num_workers):
            tg.create_task(worker_fn(i, queue))


async def streaming_processing():
    num_workers = 10
    num_tasks = 100

    async def worker_fn(worker_id: int, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        while True:
            delay = await input_queue.get()
            try:
                await asyncio.sleep(delay)
                await output_queue.put(f"[worker {worker_id}] worked for {delay:.2f}s")
            finally:
                input_queue.task_done()

    input_queue = asyncio.Queue()
    for _ in range(num_tasks):
        await input_queue.put(random.random())

    output_queue = asyncio.Queue()

    workers = [asyncio.create_task(worker_fn(i, input_queue, output_queue)) for i in range(num_workers)]

    # get output immediately when it is finished
    for _ in range(num_tasks):
        output = await output_queue.get()
        print(output)

    # cancel all workers
    for worker in workers:
        worker.cancel()

    # wait for cancellation
    await asyncio.gather(*workers, return_exceptions=True)


async def sync_to_async():
    def sync_task(delay):
        print("Starting a slow sync task...")
        time.sleep(delay)  # Simulating a long task
        print("Finished the slow task.")
        return f"Done within {delay} seconds."

    delay = 5
    loop = asyncio.get_running_loop()
    ret = await loop.run_in_executor(None, sync_task, delay)
    print(ret)


async def concurrent_to_async():
    def thread_worker(fut: concurrent.futures.Future, workload: float):
        print(f"Thread started {workload=}")
        time.sleep(workload)
        fut.set_result(workload)
        print("Thread finished")

    fut = concurrent.futures.Future()
    threading.Thread(target=thread_worker, kwargs=dict(fut=fut, workload=3), daemon=True).start()

    # chain concurrent future with asyncio future, so that we can interact with threads within coroutines
    print("Awaiting future")
    result = await asyncio.wrap_future(fut)
    print(f"Future done: {result=}")


async def await_with_timeout():
    async def sleep_5():
        logging.info("sleep_5 started")
        await asyncio.sleep(5)
        logging.info("sleep_5 finished")  # will not print

    try:
        await asyncio.wait_for(sleep_5(), timeout=3)
    except TimeoutError as e:
        logging.error(f"{e!r}")
    else:
        assert False

    await asyncio.sleep(3)


async def async_ray_task():
    # https://docs.ray.io/en/latest/ray-core/actors/async_api.html
    import ray

    ray.init()

    @ray.remote(num_cpus=os.cpu_count() // 4)
    def remote_task(i, delay):
        time.sleep(delay)
        print(f"[task {i}] done within {delay:.3f} seconds")

    num_tasks = 100
    tasks = [remote_task.remote(i=i, delay=random.random()) for i in range(num_tasks)]
    await asyncio.gather(*tasks, return_exceptions=True)


async def async_ray_actor():
    import ray

    ray.init()

    @ray.remote(num_cpus=os.cpu_count() // 4)
    class Actor:
        async def run(self, delay):
            await asyncio.sleep(delay)
            print(f"worked for {delay} seconds")

    worker = Actor.remote()
    await asyncio.gather(worker.run.remote(3), worker.run.remote(3), return_exceptions=True)


if __name__ == "__main__":
    # asyncio.run(hello_world())
    # asyncio.run(work_concurrent())
    # asyncio.run(coroutine_pool())
    asyncio.run(coroutine_pool_with_task_group())
    # asyncio.run(streaming_processing())
    # asyncio.run(sync_to_async())
    # asyncio.run(concurrent_to_async())
    # asyncio.run(await_with_timeout())
    # asyncio.run(async_ray_task())
    # asyncio.run(async_ray_actor())
