# import xml.etree.ElementTree as ET
#
# tree: ET.ElementTree = ET.parse("test.xml")
# root = tree.getroot()
# all_name = [i for i in root.findall('./value') if i.attrib['ref'] == "rule2"]
# print(all_name)


with open("note.xml") as f:
    t = f.read()
from bs4 import BeautifulSoup

soup = BeautifulSoup(t,"lxml")  # txt is simply the a string with your XML file
pageText = soup.findAll(text=True)
# print ' '.join(pageText)
print(pageText.text)
