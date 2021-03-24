from collections.abc import Iterable

# 生成器的应用场景有哪些
# 1. 生成器，针对for循环内存不够的情况
# 1. 定义上下文管理器
# 1. 协程，改成了async和await的关键字
# 1. 配合from 用于消费子生成器并传递消息
def flatten(items):
    for x in items:
        if isinstance(x, Iterable):
            yield from flatten(x)
        else:
            yield x


items = [1, 2, 3, [4, 5, [6]]]
print(type(flatten(items)))
res = [i for i in flatten(items)]
print(res)
