from typing import List, Dict

import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.101 Safari/537.36',
    'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.6,en;q=0.4',
    'Accept-Encoding': 'gzip, deflate, sdch'
}

# 一般使用广度优先爬虫，但是这里广度优先会将原先的树结构拍扁，需要集进行大量的map映射
# def province(url: str) -> Dict:
#     session = requests.Session()
#     res: requests.models.Response = session.get(url, headers=headers)
#     res.encoding = 'gb2312'
#     soup: BeautifulSoup = BeautifulSoup(res.text, "lxml")
#
#     all_province: List[BeautifulSoup] = soup.find_all('a')[:-1]
#     return {i.text: "http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2020/" + i["href"] for i in all_province}
#
#
# def couontytr(url):
#     session = requests.Session()
#     res: requests.models.Response = session.get(url, headers=headers)
#     res.encoding = 'gb2312'
#     soup: BeautifulSoup = BeautifulSoup(res.text, "lxml")
#     all_couontytr: List[BeautifulSoup] = soup.find_all('a')[:-1]
#     return {i.text: "http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2020/" + i["href"] for i in all_couontytr}


# encoding=utf-8
from bs4 import BeautifulSoup
import socket
import re
import zlib


class MyCrawler:
    def __init__(self, seeds):
        # 初始化当前抓取的深度
        self.current_deepth = 1
        # 使用种子初始化url队列
        self.linkQuence = linkQuence()
        if isinstance(seeds, str):
            self.linkQuence.addUnvisitedUrl(seeds)
        if isinstance(seeds, list):
            for i in seeds:
                self.linkQuence.addUnvisitedUrl(i)
        print("Add the seeds url \"%s\" to the unvisited url list" % str(self.linkQuence.unVisited))

    # 抓取过程主函数
    def crawling(self, seeds, crawl_deepth):
        # 循环条件：抓取深度不超过crawl_deepth
        while self.current_deepth <= crawl_deepth:
            # 循环条件：待抓取的链接不空
            while not self.linkQuence.unVisitedUrlsEnmpy():
                # 队头url出队列
                visitUrl = self.linkQuence.unVisitedUrlDeQuence()
                print("Pop out one url \"%s\" from unvisited url list" % visitUrl)
                if visitUrl is None or visitUrl == "":
                    continue
                # 获取超链接
                links = self.getHyperLinks(visitUrl)
                print("Get %d new links" % len(links))
                # 将url放入已访问的url中
                self.linkQuence.addVisitedUrl(visitUrl)
                print("Visited url count: " + str(self.linkQuence.getVisitedUrlCount()))
                print("Visited deepth: " + str(self.current_deepth))
            # 未访问的url入列
            for link in links:
                self.linkQuence.addUnvisitedUrl(link)
            print("%d unvisited links:" % len(self.linkQuence.getUnvisitedUrl()))
            self.current_deepth += 1

    # 获取源码中得超链接
    def getHyperLinks(self, url):
        links = []
        data = self.getPageSource(url)
        if data[0] == "200":
            soup = BeautifulSoup(data[1].text, "lxml")
            a = soup.findAll("a", href=True)
            for i in a:
                if i["href"].find("http://") != -1:
                    links.append(i["href"])
        return links

    # 获取网页源码
    def getPageSource(self, url, timeout=100, coding=None):
        try:
            socket.setdefaulttimeout(timeout)
            session = requests.Session()
            response: requests.models.Response = session.get(url, headers=headers)
            response.encoding = 'gb2312'
            return ["200", response]
        except Exception as e:
            print(str(e))
            return [str(e), None]


class linkQuence:
    def __init__(self):
        # 已访问的url集合
        self.visted = []
        # 待访问的url集合
        self.unVisited = []

    # 获取访问过的url队列
    def getVisitedUrl(self):
        return self.visted

    # 获取未访问的url队列
    def getUnvisitedUrl(self):
        return self.unVisited

    # 添加到访问过得url队列中
    def addVisitedUrl(self, url):
        self.visted.append(url)

    # 移除访问过得url
    def removeVisitedUrl(self, url):
        self.visted.remove(url)

    # 未访问过得url出队列
    def unVisitedUrlDeQuence(self):
        try:
            return self.unVisited.pop()
        except:
            return None

    # 保证每个url只被访问一次
    def addUnvisitedUrl(self, url):
        if url != "" and url not in self.visted and url not in self.unVisited and url != "http://www.miibeian.gov.cn/":
            self.unVisited.insert(0, url)

    # 获得已访问的url数目
    def getVisitedUrlCount(self):
        return len(self.visted)

    # 获得未访问的url数目
    def getUnvistedUrlCount(self):
        return len(self.unVisited)

    # 判断未访问的url队列是否为空
    def unVisitedUrlsEnmpy(self):
        return len(self.unVisited) == 0


def main(seeds, crawl_deepth):
    craw = MyCrawler(seeds)
    craw.crawling(seeds, crawl_deepth)


if __name__ == "__main__":
    main(["http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2020/"], 10)

# provinceurl = 'http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2020/index.html'
# print(province(provinceurl))
# countryurl = "http://www.stats.gov.cn/tjsj/tjbz/tjyqhdmhcxhfdm/2020/11/1101.html"
# print(couontytr(countryurl))
