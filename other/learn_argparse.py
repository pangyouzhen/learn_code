import argparse

parser = argparse.ArgumentParser()


# parser.add_argument('-foo', help='foo help')


# python myprogram.py --help 最简单的使用
# 高级用法 类似linux中，参数可以随意设置，
# 我们设置的一个 命令行调用的 可以做加法 可以做除法

def sub(args):
    print(args.x / args.y)


def add(args):
    print(args.x + args.y)


subparsers = parser.add_subparsers()
parser_foo = subparsers.add_parser("sub")
parser_foo.add_argument('-x', type=int, default=1)
parser_foo.add_argument('-y', type=float)
# 绑定对应的函数
parser_foo.set_defaults(func=sub)

parser_foo = subparsers.add_parser("add")
parser_foo.add_argument('-x', type=int, default=1)
parser_foo.add_argument('-y', type=float)
parser_foo.set_defaults(func=add)

args = parser.parse_args()
# 执行函数的功能
args.func(args)
