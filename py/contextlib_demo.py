from contextlib import contextmanager


def demo():
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


def nested_context():
    def ctx_a():
        print("preparing ctx a")
        try:
            yield
        finally:
            print("cleaning up ctx a")

    @contextmanager
    def ctx_b():
        print("preparing ctx b")
        try:
            yield
        finally:
            print("cleaning up ctx b")

    @contextmanager
    def switch_ctx(ctx_type="a", enabled=True):
        if not enabled:
            yield
            return

        if ctx_type == "a":
            # generator delegation
            yield from ctx_a()
        else:
            # or use another context manager
            with ctx_b():
                yield

    print("=" * 50)
    with switch_ctx(ctx_type="a"), switch_ctx(ctx_type="b"):
        print(f"work within ctx")


def reuse_context():
    @contextmanager
    def simple_context():
        print("preparing context")
        try:
            yield
        finally:
            print("cleaning up context")

    print("=" * 50)
    ctx = simple_context()
    with ctx:
        print("working")

    try:
        # cannot reuse the same context manager instance
        with ctx:
            print("working")
    except Exception as e:
        print(f"Reuse context error: {e}")


if __name__ == "__main__":
    demo()
    nested_context()
    reuse_context()
