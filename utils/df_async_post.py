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


class AsyncDf:
    """
    针对dataframe的异步请求封装
    """

    def __init__(self, df: pd.DataFrame, url: str, data: str, df_response: str, df_request_name: List[str],
                 sema: int = 30, headers=None, method="post", **kwargs):
        """
        url:  url
        df_request_name: 请求列的列名称
        df_response： 保存在df的响应列的列名称
        """
        self.url = url
        if isinstance(data, str):
            self.data = data
        else:
            assert "输入data不是str的实例"
        self.df_response = df_response
        if headers is None:
            self.headers = {"Content-Type": "application/json"}
        else:
            self.headers = headers
        self.sema = asyncio.BoundedSemaphore(sema)
        self.df = df
        self.df_request_name = df_request_name
        self.method = method

    @logger.catch()
    async def process_url(self, ind_query: Tuple) -> None:
        """
        :type ind_query: 元组，第一个为index，第二个为需要格式化的内容，
        只传入单个内容：（0,"今天天气很好啊"）
        :rtype: object
        """
        ind, rows = ind_query
        content = tuple(rows[self.df_request_name].tolist())
        data = self.data % content
        with (await self.sema):
            async with aiohttp.ClientSession() as session:
                resp = await session.post(self.url, json=json.loads(data), headers=self.headers)
                await asyncio.sleep(0.5)
                self.df.loc[ind, self.df_response] = await resp.text()

    # json.loads(data)  就是将 str转化为字典，post 的参数中json 需要的是一个字典
    # postman 中转化出来的是 str 需要改为data = data.encode("utf-8")
    # a = json.loads("...
    # a.pop("preProcess")
    # a["data"] = a["data"].strip("\"")

    async def gather_data(self):
        await asyncio.gather(
            *[self.process_url((ind, row)) for ind, row in self.df.iterrows()])

    def run(self) -> pd.DataFrame:
        start_time = datetime.datetime.now()
        print(f"start_time is {start_time}")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.gather_data())
        loop.close()
        end_time = datetime.datetime.now()
        print(f"end_time is {end_time}")
        last_time = end_time - start_time
        print(f"last_time is {last_time}")
        return self.df

    def __call__(self, *args, **kwargs):
        return self.run()

    @classmethod
    def from_postman_curl(cls, df, curl_cmd, df_response, df_request_name, sema=30):
        """
        inputs: POSTMAN - curl
        """
        import uncurl
        curl_cmd = curl_cmd.replace(" -L", "")
        curl_cmd = curl_cmd.replace("--data-raw", "-d")
        context = uncurl.parse_context(curl_cmd)
        context_kwargs = context._asdict()
        return AsyncDf(df=df, df_response=df_response, df_request_name=df_request_name, sema=sema, **context_kwargs)


if __name__ == '__main__':
    rand_num = np.random.randint(500, size=(10000, 3))
    df = pd.DataFrame(rand_num)
    df.columns = ["user1", "user2", "user3"]
    data = """{\"username\": \"%s\",\"age\": \"%s\"}"""
    df_request_name = ["user1", "user2"]
    # async_df = AsyncDf(df=df, url="http://127.0.0.1:8082/reader", data=data, df_response="reponse",
    #                    df_request_name=df_request_name, sema=100)
    # df = async_df()
    # print(df[:5])
    curl_cmd = """
    curl -L -X POST 'http://127.0.0.1:8082/reader' \
    -H 'Content-Type: application/json' \
    --data-raw '{"username":"%s","age":"%s"}'
    """
    #
    async_df = AsyncDf.from_postman_curl(df, curl_cmd, "response", df_request_name=["user1", "user2"], sema=100)
    df = async_df()
    print(df[:5])
