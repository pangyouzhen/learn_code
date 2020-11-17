class A(object):
    bar = 1


if __name__ == '__main__':
    a = A()
    print(a.bar)
    getattr(a, "bar")
    setattr(a, "bar2", "v")
    print(a.bar2)
    getattr(a, "bar2")
    a.bar3 = 6
    print(a.bar3)
    getattr(a, "bar3")
