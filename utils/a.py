from typing import List


class Child:
    name: str
    age: int
    toy: str

    def __init__(self, name: str, age: int, toy: str) -> None:
        self.name = name
        self.age = age
        self.toy = toy


class A:
    name: str
    age: int
    car: None
    children: List[Child]

    def __init__(self, name: str, age: int, car: None, children: List[Child]) -> None:
        self.name = name
        self.age = age
        self.car = car
        self.children = children
