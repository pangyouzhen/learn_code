class Tag:
    def __init__(self):
        self.change = {'python': 'This is python'}

    def __getitem__(self, item):
        #  __getitem__ 这个方法是实现 字典根据key 获取值的
        print('这个方法被调用')
        return self.change[item]

    #
    # def __new__(cls, *args, **kwargs):
    #     pass

    #  实例直接传参数
    def __call__(self, *args, **kwargs):
        pass

    def __len__(self):
        pass

    def __getattr__(self, item):
        pass


a = Tag()
print(a['python'])
