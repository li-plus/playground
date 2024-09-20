import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

# patch envvar
print("===== patch.dict =====")
print(f"before patch: FOO={os.getenv('FOO')}")
print(f"before patch: PATH={os.getenv('PATH')}")
with patch.dict(os.environ, {"FOO": "BAR", "PATH": "MY_PATH"}):
    print(f"patched FOO={os.getenv('FOO')}")
    print(f"patched PATH={os.getenv('PATH')}")
print(f"after patch: FOO={os.getenv('FOO')}")
print(f"after patch: PATH={os.getenv('PATH')}")

# patch object
print("===== patch.object =====")


@dataclass
class Person:
    name: str
    age: int


person = Person("Jason", 22)
print(f"before patch: {person=}")
with patch.object(person, "age", 33):
    print(f"patched: {person=}")
print(f"after patch: {person=}")

# patch
print("===== patch =====")


class Obj:
    def __init__(self) -> None:
        self.val1 = 1
        self.val2 = 2

    def foo(self):
        return "foo"

    def bar(self):
        return "bar"


with patch("__main__.Obj") as mock_obj:
    instance = mock_obj.return_value
    instance.foo.return_value = "patched_foo"
    obj = Obj()
    print(f"{obj.val1=} {obj.val2=} {obj.foo()=} {obj.bar()=}")

# patch.multiple
obj = Obj()
with patch.multiple(obj, val1=11, bar=MagicMock(return_value="bar_v2")):
    print(f"{obj.val1=} {obj.val2=} {obj.foo()=} {obj.bar()=}")
