# ref: https://tenacity.readthedocs.io/en/latest/

import random

from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, wait_random


@retry
def do_something_unreliable():
    if random.random() < 0.9:
        print("Run failed")
        raise RuntimeError("runtime error")
    return "Run success"


@retry(stop=stop_after_attempt(7))
def stop_after_7_attempts():
    print("Stopping after 7 attempts")
    raise Exception


@retry(stop=stop_after_delay(2))
def stop_after_2_s():
    print("Stopping after 2 seconds")
    raise Exception


@retry(stop=(stop_after_delay(10) | stop_after_attempt(5)))
def stop_after_10_s_or_5_retries():
    print("Stopping after 10 seconds or 5 retries")
    raise Exception


@retry(wait=wait_fixed(2))
def wait_2_s():
    print("Wait 2 second between retries")
    raise Exception


@retry(wait=wait_random(min=1, max=2))
def wait_random_1_to_2_s():
    print("Randomly wait 1 to 2 seconds between retries")
    raise Exception


if __name__ == "__main__":
    # print(do_something_unreliable())
    # stop_after_7_attempts()
    # stop_after_2_s()
    # stop_after_10_s_or_5_retries()
    # wait_2_s()
    wait_random_1_to_2_s()
