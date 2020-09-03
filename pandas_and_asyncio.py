import pandas as pd
import asyncio
import aiohttp
import json

sema = asyncio.BoundedSemaphore(10)
df = pd.read_excel("./df.xlsx")
df["result"] = None


# 限制线程个数，防止崩溃

async def process_url(df: pd.DataFrame, query: str) -> None:
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
                df.loc[df['问题'] == query, 'result'] = [html]
        except Exception as e:
            print(e)


async def main(df: pd.DataFrame) -> None:
    await asyncio.gather(*[process_url(df, query) for query in df['问题']])


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(df))
    loop.close()