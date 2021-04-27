# 对于io操作来说，多线程和多进程性能差别不大
# 使用多线程
# 1. 通过Thread类实例化
import threading
import time


def get_detail_html(url: str):
    print("get detail html started")
    time.sleep(2)
    print("get detail html end")


def get_detail_url(url: str):
    print("get detail url started")
    time.sleep(2)
    print("get detail url end")


# if __name__ == '__main__':
#     thread1 = threading.Thread(target=get_detail_html, args=("",))
#     thread2 = threading.Thread(target=get_detail_url, args=("",))
#     start_time = time.time()
#     thread1.start()
#     thread2.start()
#     # 还有一个主线程
#     print("last time ", time.time() - start_time)
#     print("finish")
#  1. 主线程退出，但是子线程未退出(上面程序)，需要等待子线程执行完毕
#  2. 主线程退出，如果需要子线程也退出的情况   thread2.setDaemon(True)。如果需要针对多个只设置了一个是setDaemon(True)，需要比较时间
#  3. 子线程执行完毕再执行主线程。start 之后增加 join命令即可


# 使用多线程
# 2.通过集成Thread来实现多线程
class GetDetailHtml(threading.Thread):
    # 类似java的使用
    def __init__(self, name):
        super().__init__(name)

    def run(self):
        print("get detail html started")
        time.sleep(2)
        print("get detail html end")


class GetDetailUrl(threading.Thread):
    def __init__(self, name):
        super().__init__(name)

    def run(self):
        print("get detail url started")
        time.sleep(2)
        print("get detail url end")


if __name__ == '__main__':
    thread1 = GetDetailHtml("get_detail_html")
    thread2 = GetDetailUrl("get_detail_url")
    start_time = time.time()
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    # 还有一个主线程
    print("last time ", time.time() - start_time)
    print("finish")
