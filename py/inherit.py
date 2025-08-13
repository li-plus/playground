class Base:
    def __init__(self):
        print("creating Base")


class Derived(Base):
    pass


a = Derived()  # will print message from Base.__init__
