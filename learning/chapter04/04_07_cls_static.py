class Date:
    def __init__(self, year: int, month: int, day: int):
        self.year = year
        self.month = month
        self.day = day

    def __str__(self):
        return "{}-{}-{}".format(self.year, self.month, self.day)

    # @classmethod 应用场景：用作 这个类的初始化（java的不同构造方法，java中可以有多个构造函数
    # 这种初始化不同于__init__，如果我想初始化date类，但是字符串是"2020/12/22"
    @classmethod
    def create_from_str(cls, dates: str):
        year, month, day = dates.split("/")
        return cls(year, month, day)

    # 静态方法的适合他的应用场景之一是校验传入值。
    @staticmethod
    def var_str(date_str: str):
        year, month, day = tuple(date_str.split("-"))
        if 0 < int(year) and 0 < int(month) <= 12 and 0 < int(day) < 32:
            return True
        else:
            return False


if __name__ == '__main__':
    date = Date(2020, 12, 21)
    print(date)
    date1 = Date.create_from_str("2020/12/22")
    print(date1)
