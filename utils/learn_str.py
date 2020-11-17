# 文件编码

import chardet
from pathlib import Path

file_ = Path("../full_data/train.csv")
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

tree: ET.ElementTree = ET.parse("test.xml")
root = tree.getroot()
all_name = [i.attrib['name'] for i in root.findall('./')]
print(all_name)

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
