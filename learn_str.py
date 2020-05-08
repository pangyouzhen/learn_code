# 文件编码

import chardet
from pathlib import Path

file_ = Path("./任务要求--第一课时.txt")
raw = file_.read_bytes()
encoding = chardet.detect(raw)['encoding']
print(encoding)

# 单个文件转换编码
# iconv - f GBK - t UTF-8 file1 - ofile2
# 多个文件 默认是default文件夹下面
#  find default -type d -exec mkdir -p utf/{} \;
# find default -type f -exec iconv -f GBK -t UTF-8 {} -o utf/{} \;


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
