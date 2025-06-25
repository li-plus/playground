import logging
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def simple_thread():
    def worker(sec):
        time.sleep(sec)
        print(f"Task finished within {sec} seconds")

    threads = [threading.Thread(target=worker, args=(i + 1,)) for i in range(3)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()


def thread_with_lock():
    def worker(i):
        logging.info(f"worker {i} before critical section")
        with lock:
            logging.info(f"worker {i} entered critical section")
            time.sleep(1)
        logging.info(f"worker {i} exited critical section")

    lock = threading.Lock()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()


def producer_consumer():
    class AtomicCounter:
        def __init__(self, value: int = 0):
            self._value = value
            self._lock = threading.Lock()

        def increment(self, n: int = 1):
            with self._lock:
                self._value += n
                return self._value

    def producer():
        for _ in range(10):
            sec = random.random()
            time.sleep(random.random())
            task_id = task_counter.increment()
            q.put(task_id)
            print(f"Produced task {task_id} within {sec:.2f} seconds")

    def consumer():
        while True:
            try:
                task_id = q.get(timeout=3)
            except queue.Empty:
                break
            sec = random.random()
            time.sleep(sec)
            print(f"Consumed task {task_id} within {sec:.2f} seconds")

    q = queue.Queue(maxsize=10)  # read queue.Queue source code to understand threading.Condition!
    task_counter = AtomicCounter()

    producers = [threading.Thread(target=producer) for _ in range(10)]
    consumers = [threading.Thread(target=consumer) for _ in range(10)]

    for t in chain(producers, consumers):
        t.start()

    for t in chain(producers, consumers):
        t.join()


def thread_pool():
    def worker(task_id: int):
        sec = random.random()
        time.sleep(sec)
        return f"Task {task_id} finished within {sec:.2f} seconds"

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(100)]
        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    # simple_thread()
    thread_with_lock()
    # producer_consumer()
    # thread_pool()
