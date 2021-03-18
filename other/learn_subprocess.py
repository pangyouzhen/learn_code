# https://www.cnblogs.com/vamei/archive/2012/09/23/2698014.html

import subprocess

# https://docs.python.org/zh-cn/3.9/library/subprocess.html#call-function-trio
ls = "ls -alh"
child = subprocess.Popen(ls.split(), stdout=subprocess.PIPE)
wc = "wc"
child2 = subprocess.Popen(wc, stdin=child.stdout, stdout=subprocess.PIPE)
# commit 用于继续输入的情况
out, code = child2.communicate("-l ~/oh-my-tuna.py")
print(out)
print("--------------")
# call 等是旧的api,现在统一使用run函数，但是capture_output=True，3.6不支持， 看官方文档
ll = "ls -alh"
r = subprocess.run(ll.split())
print(r.stdout)
