from typing import Tuple

import pandas as pd
import asyncio
import aiohttp
import json
from loguru import logger

logger.add("./a.log")


class DfAsyncPost:
    """
    针对dataframe的异步post请求
    """

    def __init__(self, url, payload, df_request_name, df_response, headers=None, sema=5):
        """
        url:  url
        payload : post请求的内容
        df_request_name: 请求列的列名称
        df_response： 保存在df的响应列的列名称
        :rtype:
        """
        self.url = url
        self.payload = payload
        self.df_request_name = df_request_name
        self.df_response = df_response
        if headers:
            self.headers = headers
            self.headers_json = False
        else:
            self.headers = {"Content-Type": "application/json"}
            self.headers_json = True
        self.sema = asyncio.BoundedSemaphore(sema)

    @logger.catch()
    async def process_url(self, df: pd.DataFrame, ind_query: Tuple) -> None:
        ind = ind_query[0]
        query = ind_query[1]
        payload = self.payload % query
        with (await self.sema):
            try:
                async with aiohttp.ClientSession() as session:
                    if self.headers_json:
                        resp = await session.post(self.url, json=json.loads(payload), headers=self.headers)
                    else:
                        resp = await session.post(self.url, data=payload.encode("utf-8"), headers=self.headers)
                    await asyncio.sleep(0.5)
                    df.loc[ind, self.df_response] = await resp.text()
            except Exception as e:
                logger.error(e)
                raise

    # json.loads(payload)  就是将 str转化为字典，post 的参数中json 需要的是一个字典
    # postman 中转化出来的是 str 需要改为data = payload.encode("utf-8")
    # a = json.loads("...
    # a.pop("preProcess")
    # a["data"] = a["data"].strip("\"")

    async def main(self, df: pd.DataFrame) -> None:
        await asyncio.gather(
            *[self.process_url(df, (ind, query)) for ind, query in enumerate(df[self.df_request_name])])

    def run(self, df):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.main(df))
        loop.close()


if __name__ == '__main__':
    pass
# python 3.7+
# asyncio.run(main(df))
