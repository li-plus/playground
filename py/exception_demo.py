def simple_try():
    try:
        result = 10 / 0
    except Exception as e:
        print(f"Failed: {e}")
    else:
        print(f"Success: {result}")
    finally:
        print("Execution completed.")


def try_with_continue():
    for i in range(5):
        try:
            assert i % 2 == 0, f"{i} is not even"
            continue
        except Exception as e:
            print(f"Failed: {e}")
        else:
            print(f"Success: {i}")  # unreachable
        finally:
            print("Execution completed.")  # always reachable


if __name__ == "__main__":
    simple_try()
    try_with_continue()
