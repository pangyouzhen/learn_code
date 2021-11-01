import re
from collections import defaultdict
from typing import List

import pandas as pd
from bs4 import BeautifulSoup, Tag

with open("./test.xml", encoding="utf-8") as f:
    t = f.read()
    t = re.sub("<!\[CDATA\[(.*?)]]>", "\g<1>", t)

soup: BeautifulSoup = BeautifulSoup(t, "lxml")

txt: Tag = soup.find("instantiation").find("instance-list")

df_list = defaultdict(list)
for i in txt.contents:
    if isinstance(i, Tag):
        # ReasonCode
        ReasonCode = [t.text.strip() for t in i.find_all("instance", {"ref": "ReasonCode"})]
        df_list["规则编号"].append(",".join(ReasonCode))
        # EntityProperty
        entity_property: List[Tag] = i.find_all("instance", {"ref": "EntityProperty"})
        property_list = []
        logic_list = []
        for j in entity_property:
            # property
            pros = j.find("instance", {"ref": "property"})
            # if len(property_) != 1:
            pro_text = pros.text.strip()
            property_list.append(pro_text)
            # print(pro.text.strip())
            # 操作符
            OperatorListVH = j.parent.find("instance", {"ref": "OperatorListVH"})
            # 操作值
            OperatorList_value = j.parent.find("instance-selection", {"ref": "Value"})
            opr = OperatorListVH.text.strip()
            opr_value = OperatorList_value.text.strip()
            logic = pro_text + opr + opr_value
            # print(logic)
            logic_list.append(logic)
        #  其他条件
        other_conditions = i.find_all("instance", {"ref": "Property"})
        if other_conditions:
            for i in other_conditions:
                conditions_text = i.text.strip()
                Operator = i.parent.find("instance", {"ref": "Operator"})
                Opr = Operator.text.strip()
                val = i.parent.find("instance", {"ref": "Value"})
                Opr_val = val.text.strip()
                logic = conditions_text + Opr + Opr_val
                # print(logic)
                logic_list.append(logic)
        df_list["property"].append(",".join(property_list))
        df_list["判断逻辑"].append(",".join(logic_list))
        print("------------------")
#  学到的。使用pycharm 进行xml元素的层级定位
#  使用pycharm debug模式看某个函数的各种方法/函数的结果，望文生义和debug猜测
df = pd.DataFrame(df_list)
print(df)
# df.to_excel("./resutl.xlsx", encoding="utf-8", index=False)
