from collections.abc import *


class A:
    def __next__(self):
        return 1


class B:
    def __iter__(self):
        return A()


a = A()
b = B()
print(isinstance(a, Iterable))
print(isinstance(a, Iterator))
# 只要实现了__iter__ 就是Iterable
# 只有同时实现了__next__,__iter___才是Iterator
print(isinstance(b, Iterable))
print(isinstance(b, Iterator))

print("-------------------------------------")


class Myrange:
    def __init__(self, end):
        self.start = 0
        self.end = end

    def __next__(self):
        if self.start < self.end:
            curr = self.start
            self.start += 1
            return curr
        else:
            raise StopIteration

    def __iter__(self):
        return self


myrange = Myrange(10)
print(isinstance(myrange, Iterable))
print(isinstance(myrange, Iterator))
print(next(myrange))
print("----")
for i in myrange:
    print(i)


# 调用的两种方法 for,next

# return self 是什么意思
class AA:
    def __init__(self):
        self.c = 0

    def count(self):
        self.c += 1
        return self


#  返回这个实例本身
aa = AA()
print(isinstance(aa.count(), AA))
aa.count().count()
print(aa.c)
