import abc


class Base(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def func_a(self, data: str) -> str:
        pass

    @abc.abstractmethod
    def func_b(self, data: str) -> str:
        pass

    def func_c(self, data: str) -> str:
        pass


class Sub(Base):

    def func_a(self, data: str) -> str:
        return "a"

    def func_b(self, data: str) -> str:
        return "b"

    def func(self):
        return "a"

# ABCMeta将原始的类变成一个虚类，父类用@abc.abstractmethod修饰的方法，在子类中必须实现，否则就会报错
# NotImplementedError 不是这样，这个只是调用这个方法时才会报错
if __name__ == '__main__':
    s = Sub()
    print(s.func())
