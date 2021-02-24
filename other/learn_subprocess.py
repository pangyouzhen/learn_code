# https://www.cnblogs.com/vamei/archive/2012/09/23/2698014.html

import subprocess

ls = "ls -alh"
child = subprocess.Popen(ls.split(), stdout=subprocess.PIPE)
wc = "wc"
child2 = subprocess.Popen(wc, stdin=child.stdout, stdout=subprocess.PIPE)
# commit 用于继续输入的情况,todo
out, code = child2.communicate("-l ~/oh-my-tuna.py")
print(out)
print("--------------")
# 没有交互的 使用subprocess.call,返回的结果是整数，如果是0，正确执行，否则执行error
ll = "ls -alh"
r = subprocess.call(ll.split())
print(r)
