import json
import time
import numpy as np


def not_a_generator():
    result = []
    for i in range(2000):
        result.append(i ** 2)
    return result


def is_a_generator():
    for i in range(2000):
        yield i ** 2


def is_a_generator_ls():
    return (i ** 2 for i in range(2000))


n = time.time()
print(not_a_generator())
n1 = time.time()
print(is_a_generator_ls())
print([i for i in is_a_generator_ls()])
print(n1 - n)
print([i for i in is_a_generator()])
print(time.time() - n1)


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


def flat(x):
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in flat(value):
                k = f'{key}_{k}'
                yield (k, v)
        else:
            yield (key, value)


if __name__ == '__main__':
    nest_dict = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3,
            'e': {'f': 4}
        },
        'g': {'h': 5},
        'i': 6,
        'j': {'k': {'l': {'m': 8}}}
    }
    print({k: v for k, v in flat(nest_dict)})
