import argparse
import subprocess
from pathlib import Path

# 批量解压的tgz脚本

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="./")
args = parser.parse_args()
print(args.dir)
file = Path(args.dir)

tar_cmd = "tar -xvf %s -C %s"
mkdir_cmd = "mkdir %s"
for i in file.glob("*tgz"):
    subprocess.run((mkdir_cmd % i.stem).split())
    subprocess.run((tar_cmd % (i, i.stem)).split())
