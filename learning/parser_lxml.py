# pip install beautifulsoup4
# pip install lxml
from bs4 import BeautifulSoup, ResultSet

with open("xml2.xml", encoding="utf-8") as f:
    t = f.read()
soup: BeautifulSoup = BeautifulSoup(t, "lxml")
pageText: ResultSet = soup.findAll("instance")
for page in pageText:
    print(page["ref"])
    print(page.text.strip())
    print("---------")
# print ' '.join(pageText)
print("finish")
