import execjs
import requests




with open(r'/home/pang/Downloads/aes.min.js', 'r', encoding='utf-8') as f:
    js = f.read()

with open('/data/project/learn_code/other/encrty.js', 'r', encoding='utf-8') as fs:
    js_file = fs.read()

ct = execjs.compile(js, cwd=r'/usr/lib/node_modules')
js_load = execjs.compile(js_file, cwd=r'/usr/lib/node_modules')
uname = js_load.call('thsencrypt.encode', 'pangtong126')
passwd = js_load.call('thsencrypt.encode', 'thsmmxztxq2011')
v = ct.call("v")
print(uname)
print(ct.call("v"))

url = "http://upass.iwencai.com/login/dologinreturnjson2"

payload = {
    "uname": "%s" % uname,
    "passwd": "%s" % passwd,
    "captcha": "%s",
    "longLogin": "on",
    "rsa_version": "default_4",
    "source": "iwc_web"
}
headers = {
    'POST': ' /login/dologinreturnjson2 HTTP/1.1',
    'Host': ' upass.iwencai.com',
    'Connection': ' keep-alive',
    "hexin-v": "%s" % v,
    'Content-Length': ' 450',
    'Accept': ' application/json, text/javascript, */*; q=0.01',
    'X-Requested-With': ' XMLHttpRequest',
    'User-Agent': ' Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36',
    'Content-Type': ' application/x-www-form-urlencoded; charset=UTF-8',
    'Origin': ' http://upass.iwencai.com',
    'Referer': ' http://upass.iwencai.com/login?act=loginByIframe&view=public&source=iwc_web&main=7&detail=3&redir=http%3A%2F%2Fwww.iwencai.com%2Funifiedwap%2Flogin-sso.html',
    'Accept-Encoding': ' gzip, deflate',
    'Accept-Language': ' zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    'Cookie': ' PHPSESSID=knstqaqbslsmns9ql3cdbqd91a91ee0n; cid=0bdf07ddd90f80d8706fd264827abd8a1618119272; v=%s' % v
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
