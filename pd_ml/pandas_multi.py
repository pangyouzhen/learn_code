import pandas as pd
import psutil
from pathlib import Path
from multiprocessing import Pool
from datetime import datetime

now = datetime.now()
now_str = now.strftime("%Y-%m-%d")

path = Path("./")


# 其他[ray] [modin] 对性能没有明显提升，2021-01-27
def get_data(file):
    content = []
    try:
        df = pd.read_csv(file)
        content = df["聊天内容"].tolist()
    except Exception as e:
        print(e)

    with open("/tmp/" + "%s_%s.txt" % (now_str, file)) as f:
        f.write("".join(content))


def main():
    p = Pool(psutil.cpu_count() - 1)
    for i in path.glob("*.csv"):
        p.apply_async(get_data, args=(i.name,))


if __name__ == '__main__':
    main()
