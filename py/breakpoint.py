def foo():
    s = 0
    for i in range(10):
        if i == 5:
            breakpoint()  # equivalent to: import pdb; pdb.set_trace()
        s += i


foo()
