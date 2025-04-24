"""
docs: https://docs.python.org/3/library/asyncio-task.html
tutorial: https://medium.com/@moraneus/mastering-pythons-asyncio-a-practical-guide-0a673265cf04
"""

import asyncio
import random
import time


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
            print(f"[worker {worker_id}] sleeping for {delay:.3f}s")
            try:
                await asyncio.sleep(delay)
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


if __name__ == "__main__":
    # asyncio.run(hello_world())
    # asyncio.run(work_concurrent())
    # asyncio.run(coroutine_pool())
    asyncio.run(streaming_processing())
