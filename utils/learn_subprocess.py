import subprocess
import datetime
import time

add_cmd = "git add ."
commit_cmd = "git commit -m fixed"
push_cmd = "git push origin master"

for i in range(10):
    print(f"进行第{i}次提交")
    subprocess.run(add_cmd.split())
    subprocess.run(commit_cmd.split())
    res = subprocess.run(push_cmd.split())
    if res != 0:
        print(res)
        time.sleep(30)
    else:
        break
