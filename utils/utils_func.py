def flat(x):
    for key, value in x.items():
        if isinstance(value, dict):
            for k, v in flat(value):
                k = f'{key}_{k}'
                yield (k, v)
        else:
            yield (key, value)


#  用于将json 扁平化, 获取相应的key之后进行对比
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


if __name__ == '__main__':
    nest_dict = {
        'a': 1,
        'b': [{
            'c': 2,
            'd': 3,
            'e': {'f': 4}
        },
            {
                'c': 2,
                'd': 3,
                'e': {'f': 4}
            }
        ],
        'g': {'h': 5},
        'i': 6,
        'j': {'k': {'l': {'m': 8}}}
    }
    print({k: v for k, v in flat(nest_dict)})
    print(flatten_json(nest_dict))
