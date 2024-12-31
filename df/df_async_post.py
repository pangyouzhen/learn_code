import asyncio
import datetime
from typing import Tuple, List

import aiohttp
import pandas as pd
from loguru import logger
import uncurl

def run_time_wraps(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        print(f"start_time is {start_time}")
        res = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        print(f"end_time is {end_time}")
        last_time = end_time - start_time
        print(f"last_time is {last_time}")
        return res
    return wrapper

class AsyncDf:
    """针对dataframe的异步请求封装，使用单个session"""

    def __init__(self, df: pd.DataFrame, df_response: str, df_request_name: List[str],
                 sema: int = 30, **kwargs):
        self.data = kwargs.pop("data")
        self.df_response = df_response
        self.df_request_name = df_request_name
        self.kwargs = kwargs
        self.url = self.kwargs.pop("url")
        self.method = self.kwargs.pop("method")
        self.sema = asyncio.BoundedSemaphore(sema)
        self.df = df.copy() # 创建副本，避免修改原始DataFrame
        self.session = None # 初始化session为None

    @logger.catch
    async def _process_url(self, ind_query: Tuple) -> None:
        ind, row = ind_query
        content = tuple(row[self.df_request_name].tolist())
        data = self.data % content
        async with self.sema:
            try:
                resp = await self.session.request(self.method, self.url, data=data, **self.kwargs) # 使用self.session
                self.df.loc[ind, self.df_response] = await resp.text()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e: # 添加TimeoutError处理
                logger.error(f"Error processing row {ind}: {e}")
                self.df.loc[ind, self.df_response] = f"Error: {e}" # 记录错误信息到DataFrame
            except Exception as e:
                logger.exception(f"Unexpected error processing row {ind}: {e}") # 记录完整异常信息
                self.df.loc[ind, self.df_response] = f"Unexpected Error: {e}"

    async def _gather_data(self):
        tasks = [self._process_url((ind, row)) for ind, row in self.df.iterrows()]
        await asyncio.gather(*tasks, return_exceptions=True)

    @run_time_wraps
    def run(self) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        async def run_async(): # 定义一个内部async函数
            async with aiohttp.ClientSession() as session: # 在这里创建session
                self.session = session # 将session赋值给self.session
                await self._gather_data()
                self.session = None # 请求完成后将session置为None
        try:
            loop.run_until_complete(run_async())
        except asyncio.CancelledError:
            logger.warning("Operation cancelled.")
        finally:
            loop.close()
        return self.df

    @classmethod
    def from_curl(cls, df, curl_cmd, df_response, df_request_name, sema=30):
        curl_cmd = curl_cmd.replace(" -L", "").replace("--data-raw", "-d")
        context = uncurl.parse_context(curl_cmd)
        context_kwargs = context._asdict()
        return cls(df=df, df_response=df_response, df_request_name=df_request_name, sema=sema, **context_kwargs)

if __name__ == '__main__':
    df = pd.DataFrame({'query': ['你好', '世界', 'Python', '编程']})
    curl_cmd = """
    curl -X POST 'https://fanyi.baidu.com/langdetect' \
    -H 'Content-Type: application/x-www-form-urlencoded; charset=UTF-8' \
    --data-raw 'query=%s'
    """
    async_df = AsyncDf.from_curl(df, curl_cmd, "baidu_langs", df_request_name=["query"], sema=20)
    df = async_df()
    print(df)
    # df.to_csv("baidu_lang.csv", index=False)