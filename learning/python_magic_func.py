class Tag:
    def __init__(self):
        pass

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Tag1(Tag):
    def __init__(self, change):
        super().__init__()
        self.change = change

    # __getitem__ pytorch中的dataloader有应用
    def __getitem__(self, item):
        #  __getitem__ 这个方法是实现 字典根据key 获取值的
        return self.change[item]

    def __len__(self):
        return len(self.change)


a = Tag1({'python': 'This is python',
          "java": "this is java"})
print(a['python'])
print(a["java"])

print("-----------------------")


# 如果是self.change 是itertable 那么可以 直接进行循环
class Tag2(Tag):
    def __init__(self, change):
        super().__init__()
        self.change = change

    def __getitem__(self, item):
        #  __getitem__ 这个方法是实现 字典根据key 获取值的
        return self.change[item]

    def __len__(self):
        return len(self.change)


tag2 = Tag2(["python", "java"])

# mixin
# 多重类的组合。而不是在原始类上扩展
# 比如 bird 类和runable 类 -> 两者组合可以是鸵鸟，如果鸵鸟只继承bird，那么需要自己扩展相关的方法
# requests 包里面 的Session 就是用了mixin

# 线程池和进程池
# 线程池和进程池是一个普遍的做法
# 可以看下session中是如何进行构建线程池的
# TCP 中的五元组



for i in tag2:
    print(i)
print(tag2[:1])
print(len(tag2))
# for 调用可迭代类型，应该 按照顺序，__iter__,__getitem__
# 打印继承关系  内部的__mro__ 方法
print(Tag2.mro())
