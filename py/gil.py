# https://realpython.com/python-gil/
# The Python Global Interpreter Lock or GIL, in simple words, is a mutex (or a lock) that allows only one thread to hold the control of the Python interpreter.

import sys
import time
from multiprocessing import Pool
from threading import Thread

# reference count variable needed protection from race conditions where two threads increase or decrease its value simultaneously
a = []
b = a
print("ref count", sys.getrefcount(a))
del b
print("ref count", sys.getrefcount(a))

# single thread
COUNT = 50000000


def countdown(n):
    while n > 0:
        n -= 1


start = time.time()
countdown(COUNT)
end = time.time()

print("Time taken in seconds -", end - start)

# multi threads
t1 = Thread(target=countdown, args=(COUNT // 2,))
t2 = Thread(target=countdown, args=(COUNT // 2,))

start = time.time()
t1.start()
t2.start()
t1.join()
t2.join()
end = time.time()

print("Time taken in seconds -", end - start)

# multi processes
pool = Pool(processes=2)
start = time.time()
r1 = pool.apply_async(countdown, [COUNT // 2])
r2 = pool.apply_async(countdown, [COUNT // 2])
pool.close()
pool.join()
end = time.time()
print("Time taken in seconds -", end - start)
