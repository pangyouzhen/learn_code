from typing import Tuple

import pandas as pd
import asyncio
import aiohttp
import json
from loguru import logger

sema = asyncio.BoundedSemaphore(10)
df = pd.read_excel("./data/df.xlsx")
df["result"] = None


# 限制线程个数，防止崩溃

@logger.catch()
async def process_url(df: pd.DataFrame, ind_query: Tuple) -> None:
    ind = ind_query[0]
    query = ind_query[1]
    with (await sema):
        try:
            async with aiohttp.ClientSession() as session:
                payload = "{\"openId\":\"1111111\",\"userInput\":%s}" % query
                url = "http://127.0.0.1/8000/index"
                # 协程嵌套，只需要处理最外层协程即可fetch_async
                async with session.post(url, json=json.loads(payload)) as resp:
                    html = (await resp.text())
                    await asyncio.sleep(1)
                    # 因为这里使用到了await关键字，实现异步，所有他上面的函数体需要声明为异步async
                df.loc[ind, 'result'] = html
        except Exception as e:
            logger.error(e)


# json.loads(payload)  就是将 str转化为字典，post 的参数中json 需要的是一个字典
# postman 中转化出来的是 str 需要改为data = payload.encode("utf-8")
# a = json.loads("...")
# a.pop("preProcess")
# a["data"] = a["data"].strip("\"")

async def main(df: pd.DataFrame) -> None:
    await asyncio.gather(*[process_url(df, (ind, query)) for ind, query in enumerate(df['问题'])])


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(df))
    loop.close()
    # python 3.7+
    # asyncio.run(main(df))
