#  pip install PyExecJS
import execjs

# IIFE: Immediately Invoked Function Expression，意为立即调用的函数表达式，也就是说，声明函数的同时立即调用这个函数。
# 原始的文件中是 这种写法
# iwencai 登陆 uname 和passwd js逆向
with open('/tmp/untitled/ll.js', 'r', encoding='utf-8') as f:
    js_file = f.read()


js_load = execjs.compile(js_file)
print("finish")