class Student(object):
    # 以元祖的方式定义能给对象添加的属性，除此之外的属性不能添加，对动态添加属性可以做出一些限制
    __slots__ = ('name', 'age')  # 用tuple定义允许绑定的属性名称


s = Student()  # 创建新的实例
s.name = 'Michael'  # 绑定属性'name'
s.age = 25  # 绑定属性'age'
s.score = 99  # 绑定属性'score'
print(s)
