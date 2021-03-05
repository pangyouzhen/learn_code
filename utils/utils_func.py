import json
import numpy as np


def parser_js(x: str, keys_: str):
    js = json.loads(x)
    k = keys_.split(".")
    try:
        for i in k:
            if i == "$":
                # 1. 仅有一个key 2. key是不确定的
                k = "".join(list(js.keys()))
                js = js[k]
                continue
            js = js[i]
    except KeyError:
        return np.NAN
    except AttributeError:
        return np.NAN
    return js
