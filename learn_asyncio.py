# import asyncio, aiohttp
# from time import time
#
# sema = asyncio.BoundedSemaphore(10)
#
#
# # 限制线程个数，防止崩溃
#
# async def fetch_async(url):
#     print(f"{url}")
#     with (await sema):
#         try:
#             async with aiohttp.ClientSession() as session:
#                 # 协程嵌套，只需要处理最外层协程即可fetch_async
#                 async with session.get(url) as resp:
#                     html = (await resp.text())
#                     # 因为这里使用到了await关键字，实现异步，所有他上面的函数体需要声明为异步async
#         except Exception as e:
#             print(e)
#     return html, url
import pandas as pd
import asyncio
import aiohttp
import json

sema = asyncio.BoundedSemaphore(10)
df = pd.read_excel("./df.xlsx")
df["result"] = None


# 限制线程个数，防止崩溃

async def process_url(df, query):
    with (await sema):
        try:
            async with aiohttp.ClientSession() as session:
                payload = "{\"openId\":\"1111111\",\"userInput\":%s}" % query
                url = "http://127.0.0.1/8000/index"
                # 协程嵌套，只需要处理最外层协程即可fetch_async
                async with session.get(url) as resp:
                    html = (await resp.text())
                    # 因为这里使用到了await关键字，实现异步，所有他上面的函数体需要声明为异步async
                df.loc[df['问题'] == query, 'result'] = [json.loads(html)]
        except Exception as e:
            print(e)


# if __name__ == '__main__':
#     start_time = time()
#     url = "http://httpbin.org/get"
#     tasks = []
#     for i in range(100):
#         tasks.append(fetch_async(url))
#     print("start fetch")
#     # tasks = [fetch_async('http://www.baidu.com/'), fetch_async('http://www.cnblogs.com/ssyfj/')]
#     event_loop = asyncio.get_event_loop()
#     results = event_loop.run_until_complete(asyncio.gather(*tasks))
#     for i in results:
#         # print(i["learn_code"])
#         print(i)
#     event_loop.close()
#     print(f'async time {time() - start_time}')
async def main(df):
    await asyncio.gather(*[process_url(df, query) for query in df['问题']])


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(df))
    loop.close()
