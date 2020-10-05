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

for i in tag2:
    print(i)
print(tag2[:1])
print(len(tag2))
# for 调用可迭代类型，应该 按照顺序，__iter__,__getitem__