from contextlib import contextmanager


@contextmanager
def ctx():
    print("preparing context")
    try:
        yield
    finally:
        print("cleaning up context")


print("=" * 50)
with ctx():
    print("do something within context")


def ctx_a():
    print("preparing ctx a")
    yield
    print("cleaning up ctx a")


def ctx_b():
    print("preparing ctx b")
    yield
    print("cleaning up ctx b")


@contextmanager
def switch_ctx(ctx_type="a", enabled=True):
    if not enabled:
        yield
        return

    # generator delegation
    if ctx_type == "a":
        yield from ctx_a()
    else:
        yield from ctx_b()


print("=" * 50)
with switch_ctx(ctx_type="b"):
    print(f"work within ctx")
