# 文件编码

import chardet
from pathlib import Path

file_ = Path("./full_data/train.csv")
raw = file_.read_bytes()
encoding = chardet.detect(raw)['encoding']
# python 3.8 高级用法
# print(f"{encoding=}")
print(encoding)

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

# XML
import xml.etree.ElementTree as ET

tree = ET.parse("test.xml")
root = tree.getroot()
all_name = [i.attrib['name'] for i in root.findall('./')]
print(all_name)
