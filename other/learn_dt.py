import arrow

print(arrow.now())
# 转化成字符串
print(arrow.now().format())
print(arrow.now().format("YYYY-MM-DD HH:mm:ss"))
# 昨天
print(arrow.now().shift(days=-1).format())
# 转化为时间戳
print(arrow.now().timestamp())
