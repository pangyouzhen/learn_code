#生成器和协程的学习
def foo():
    bar_a = yield 1
    bar_b = yield bar_a
    yield "最后一个值"


def foo2():
    yield 1
    yield 2
    yield "最后一个值"

def consumer():
    r = ''
    while True:
        n = yield r
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'

def produce(c):
    c.send(None)
    #  send None就相当于next(c)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()

# 最大的优势就是协程极高的执行效率。因为子程序切换不是线程切换，而是由程序自身控制，因此，没有线程切换的开销，和多线程比，线程数量越多，协程的性能优势就越明显。
#
# 第二大优势就是不需要多线程的锁机制，因为只有一个线程，也不存在同时写变量冲突，在协程中控制共享资源不加锁，只需要判断状态就好了，所以执行效率比多线程高很多。



if __name__ == '__main__':
    # f = foo()
    # print(next(f))
    # print(next(f))
    # print(next(f))
    # 返回结果
    # 1
    # None 注意这里为啥是None，因为1的值并没有赋值给bar_a,就已经返回了
    # 最后一个值

    c = consumer()
    produce(c)
