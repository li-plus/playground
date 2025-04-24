from contextlib import ExitStack, contextmanager


@contextmanager
def simple_context(idx):
    print(f"entering context {idx}")
    try:
        yield
    finally:
        print(f"exiting context {idx}")


print("before stack")

with ExitStack() as stack:
    for i in range(10):
        stack.enter_context(simple_context(i))

print("after stack")
