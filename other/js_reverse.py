#  pip install PyExecJS
import execjs

# iwencai 登陆 uname 和passwd js逆向
with open('./encrty.js', 'r', encoding='utf-8') as f:
    js_file = f.read()

# print(js_file)

# 加载JS文件

js_load = execjs.compile(js_file)

password = js_load.call('thsencrypt.encode', 'Steven2020')
print(password)