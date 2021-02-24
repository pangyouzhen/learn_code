import time
import random

# class ProxyProvider:
#     def __init__(self):
#         self.pool = []
#         self.last_update_time = 0
#
#     def get_proxy(self):
#         now = time.time()
#         if now - self.last_update_time > 3600 or not self.pool:
#             self.pool = self.get_all_proxies_from_redis()
#         return random.choice(self.pool)
#
#     def get_all_proxies_from_redis(self):
#         pass
from typing import List, Set


class ProxyProvider2:
    def __init__(self):
        self.pool: List = []
        self.special_ip: Set = set()
        self.last_update_time: int = 0

    @property
    def proxy(self):
        now = time.time()
        if now - self.last_update_time > 3600 or not self.pool:
            self.pool = self.get_all_proxies_from_redis()
        return random.choice(self.pool + list(self.special_ip))

    @proxy.setter
    def proxy(self, value: str) -> None:
        if not value.startswith('http'):
            proxy = f'http://{value}'
        else:
            proxy = value
        if proxy in self.special_ip:
            return
        self.special_ip.add(proxy)

    def get_all_proxies_from_redis(self) -> List:
        return ["a", ]


#    @property,@proxy.setter  将方法设置为属性，并且这个属性 需要校验或者其他的预处理操作，否则可以直接使用self 属性
if __name__ == '__main__':
    provider = ProxyProvider2()
    provider.proxy = '123.45.67.89'
    print(provider.proxy)
