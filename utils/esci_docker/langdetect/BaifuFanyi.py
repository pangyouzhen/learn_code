from typing import List
from .DetectLang import DetectLang
import requests
import json


class BaiduFanyiDetectLang(DetectLang):
    
    def __init__(self,model_path=None,**kwargs) -> None:
        # 内网不支持 baidu翻译接口
        self.module_path = model_path
        self.url = "https://fanyi.baidu.com/langdetect"
        self.header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:101.0) Gecko/20100101 Firefox/101.0',
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'X-Requested-With': 'XMLHttpRequest',
        'Origin': 'https://fanyi.baidu.com',
        'Connection': 'keep-alive',
        'Referer': 'https://fanyi.baidu.com/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Cookie': 'BAIDUID=5205CCF1BE28D5ECBFA955B10D527899:FG=1'
        }

    
    def detect_lang(self,text:List[str],batch_size) -> List[str]:
        # 
        payload = "query=I+like+you.+I+love+you"
        response = requests.request("POST", self.url, headers=self.headers, data=payload)
        print(response.text)
