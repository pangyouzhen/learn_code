# GIL
# GIL中的一个线程对应c语言的一个线程，python py编译成 字节码
# import dis
#
#
# def add_(x: int):
#     x = x + 1
#     return x
#
#
# print(dis.dis(add_))

total = 0


def add():
    global total
    for i in range(1000000):
        total += 1


def desc():
    global total
    for i in range(1000000):
        total -= 1


import threading

thread1 = threading.Thread(target=add)
thread2 = threading.Thread(target=desc)

thread1.start()
thread2.start()

thread1.join()
thread2.join()
print(total)
#  这个结果不是0，也不是一个稳定的数值，所以2点
# 1. 不是等待这个线程完全执行ok后才进行下一个线程
# 2. 线程会在某个时刻释放该锁，这个需要结合字节码来分析
# gil 会根据执行的字节码行数和时间片进行释放，在IO操作时会主动释放
