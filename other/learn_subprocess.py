# https://www.cnblogs.com/vamei/archive/2012/09/23/2698014.html

import subprocess

ls = "ls -alh"
child = subprocess.Popen(ls.split(), stdout=subprocess.PIPE)
wc = "wc"
child2 = subprocess.Popen(wc, stdin=child.stdout, stdout=subprocess.PIPE)
out, code = child2.communicate()
print(out)
