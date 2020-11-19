import json


def main():
    with open("../data/2020年最新全国行政区划json.json") as f:
        temp = f.read()
    js = json.loads(temp)
    res = []
    for i in js:
        province = i["name"]
        for area in i["areaList"]:
            city = area["name"]
            for j in area["areaList"]:
                res.append(j["name"] + "\n")
            res.append(city + "\n")
        res.append(province + "\n")
    with open("../data/addr.txt", "w", encoding="utf-8") as address:
        address.writelines(res)


if __name__ == '__main__':
    main()
