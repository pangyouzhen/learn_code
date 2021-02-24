from math import hypot, sqrt


class Vector:
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def __repr__(self):
        print("__repr___")
        return "Vector (%r,%r)" % (self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other: 'Vector'):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __abs__(self):
        # return hypot(self.x, self.y)
        return sqrt(self.x * self.x + self.y * self.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
#    如果没有定义bool方法，则默认调用len方法
