from typing import Tuple, Dict, List

import pandas as pd
import asyncio
import aiohttp
import json
from loguru import logger
import numpy as np
import datetime
import uuid


# logger.add("./a.log")


class DfAsyncPost:
    """
    针对dataframe的异步post请求
    """

    def __init__(self, url: str, payload: str, df_response: str, sema: int = 5):
        """
        url:  url
        payload : post请求的内容
        df_request_name: 请求列的列名称
        df_response： 保存在df的响应列的列名称
        :rtype:
        """
        self.url = url
        if isinstance(payload, str):
            self.payload = payload
        else:
            assert "输入payload不是str的实例"
        self.df_response = df_response
        self.headers = {"Content-Type": "application/json"}
        self.headers_json = True
        self.sema = asyncio.BoundedSemaphore(sema)

    @logger.catch()
    async def process_url(self, df: pd.DataFrame, ind_query: Tuple) -> None:
        """
        :type ind_query: 元组，第一个为index，第二个为需要格式化的内容，
        只传入单个内容：（0,"今天天气很好啊"）
        传入多个需要格式化内容比如用户id和用户内容(0,"1111111$$$今天天气很好啊")
        :rtype: object
        """
        ind = ind_query[0]
        if not ind_query[1]:
            return
        query = str(ind_query[1]).split("$$$")
        payload = self.payload % query[0]
        with (await self.sema):
            async with aiohttp.ClientSession() as session:
                resp = await session.post(self.url, json=json.loads(payload), headers=self.headers)
                await asyncio.sleep(0.5)
                df.loc[ind, self.df_response] = await resp.text()

    # json.loads(payload)  就是将 str转化为字典，post 的参数中json 需要的是一个字典
    # postman 中转化出来的是 str 需要改为data = payload.encode("utf-8")
    # a = json.loads("...
    # a.pop("preProcess")
    # a["data"] = a["data"].strip("\"")

    async def gather_data(self, df: pd.DataFrame, df_request_name: str) -> None:
        await asyncio.gather(
            *[self.process_url(df, (ind, query)) for ind, query in enumerate(df[df_request_name])])

    def run(self, df: pd.DataFrame, df_request_name: str) -> pd.DataFrame:
        start_time = datetime.datetime.now()
        print(f"start_time is {start_time}")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.gather_data(df, df_request_name))
        loop.close()
        end_time = datetime.datetime.now()
        print(f"end_time is {end_time}")
        last_time = end_time - start_time
        print(f"last_time is {last_time}")
        return df

    def get_gater_data(self, df: pd.DataFrame, df_request_name: str) -> asyncio.Future:
        return asyncio.gather(
            *[self.process_url(df, (ind, query)) for ind, query in enumerate(df[df_request_name])])

    @classmethod
    def from_postman_curl(cls, inputs, df_response):
        """
        inputs: POSTMAN - curl
        """
        try:
            import uncurl
        except:
            assert "can't find uncurl"


if __name__ == '__main__':
    rand_num = np.random.randint(500, size=(10000, 3))
    df = pd.DataFrame(rand_num)
    df.columns = ["user1", "user2", "user3"]
    payload = """{\"username\": \"%s\",\"age\": \"%s\",}"""
    df_request_name = ["user1", "user2"]
    df_async = DfAsyncPost(url="http://81.71.140.148:8082/reader", payload=payload, df_response="reponse", sema=100)
    df_async.run(df, "user1")
    print(df[:5])
    # curl_cmd = """curl --location --request POST 'http://127.0.0.1:8082/reader' \
    #             --header 'Content-Type: application/json'
    #             --data-raw '{"username":"pangyouzhen"}'"""
    # df_async = DfAsyncPost.from_postman_curl(curl_cmd, "response")
