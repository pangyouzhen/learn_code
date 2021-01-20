import psutil
from collections import namedtuple

# 获取内存信息
svmem: namedtuple = psutil.virtual_memory()
print(svmem.total)
# 获取cpu信息
cpu_num: int = psutil.cpu_count()
print(cpu_num)
print(psutil.cpu_count(logical=False))
