# 文件编码
import chardet
from pathlib import Path

file_ = Path("../full_data/train.csv")
raw = file_.read_bytes()
encoding = chardet.detect(raw)['encoding']
# python 3.8 高级用法
# print(f"{encoding=}")
# print(f"{2+1=}")
print(encoding)
# 打印当前文件的名字，可以将文件名保存成这个
print(Path(__file__).stem)
# 单个文件转换编码
# iconv - f GBK - t UTF-8 file1 - ofile2
# 多个文件 默认是default文件夹下面
#  find default -type d -exec mkdir -p utf/{} \;
# find default -type f -exec iconv -f GBK -t UTF-8 {} -o utf/{} \;

# Pathlib 常见用法
# 绝对路径
print(file_.absolute())
# 前缀和后缀,前缀不是 prefix 而是stem，
print(file_.stem)
print(file_.suffix)
# 子目录 直接 /
# q = file_/'capthcha'
# 枚举所有文件
# p.glob('**/*.py')

print("-------------------")
#### config

import configparser

cf = configparser.ConfigParser()
cf.read("db.conf")
db_host = cf.get('db', 'ip')
print(db_host)

import configparser

cf = configparser.ConfigParser()
cf.read("db.conf")

db_host = cf._sections


def fn(ip, port=9):
    return ip, port


fn(**db_host["db"])

# XML用bs4解析

import uuid

# uuid4 是基于 伪随机数的
id = str(uuid.uuid4())

# python 提取字符串中的汉字
str1 = "｛我%$是，《速$@.度\发》中 /国、人"
str2 = "[齐天大圣/孙悟空] 2016.09.17 六学家Zhang ~ 第1张.jpg"
import re

res1 = ''.join(re.findall('[\u4e00-\u9fa5]', str1))
print(res1)

# 保留字母，汉字，数字
res2 = re.sub("[^a-zA-Z0-9\u4e00-\u9fa5]", '', str2)
print(res2)

# 元组拆包 *

def f(*kw):
    print(kw)


f(["a", "b", "c"])
f(("今天",))
f("今天天气很好啊")
f(*["A", "B", "C"])
f(*"明天咋样")
f(*("明早上",))
f(*("哈哈哈哈"))

import re

phone_number = "\"phone\": \"15555555555\""
res2 = re.sub("\"phone\": \"(\d{1})\d{8}(\d{2})\"", '\g<1>********\g<2>', phone_number)
print(res2)

# 时间日期
import arrow

print(arrow.now())
# 转化成字符串
print(arrow.now().format())
print(arrow.now().format("YYYY-MM-DD HH:mm:ss"))
# 昨天
print(arrow.now().shift(days=-1).format())
# 转化为时间戳
print(arrow.now().timestamp())

import psutil
from collections import namedtuple

# 获取内存信息
svmem: namedtuple = psutil.virtual_memory()
print(svmem.total)
# 获取cpu信息
cpu_num: int = psutil.cpu_count()
print(cpu_num)
print(psutil.cpu_count(logical=False))
