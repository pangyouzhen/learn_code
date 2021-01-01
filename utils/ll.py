import requests
# requests.get()


from typing import TypeVar, Generic
from logging import Logger

T = TypeVar('T')


class LoggedVar(Generic[T]):
    def __init__(self, value: T, name: str, logger: Logger) -> None:
        self.name = name
        self.logger = logger
        self.value = value

    def set(self, new: T) -> None:
        self.log('Set ' + repr(self.value))
        self.value = new

    def get(self) -> T:
        self.log('Get ' + repr(self.value))
        return self.value

    def log(self, message: str) -> None:
        self.logger.info('%s: %s', self.name, message)


from typing import Iterable


def zero_all_vars(vars: Iterable[LoggedVar[int]]) -> int:
    a = ""
    for var in vars:
        var.set("deeef")
        return var.get()


if __name__ == '__main__':
    logger = Logger("allen_nlp")
    a = LoggedVar(1000, "a", logger)
    zero_all_vars([a])
