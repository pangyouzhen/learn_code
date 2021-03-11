import datetime
import time

now = datetime.datetime.now()
print('\n当前时间datetime：\t%s，类型：%s' % (now, type(now)))
# 当前时间str
now_str = str(now)
print('当前时间str：\t\t%s，类型：%s' % (now_str, type(now_str)))
# 当前时间时间戳
now_mktime = time.time()
print('当前时间时间戳：\t%s，类型：%s' % (now_mktime, type(now_mktime)))

# 今天、昨天、明天
today = datetime.date.today()
yesterday = today - datetime.timedelta(days=1)
tomorrow = today + datetime.timedelta(days=1)
print('\n今天:%s,昨天:%s,明天：%s' % (today, yesterday, tomorrow))

# 时间的转换 见 img/dt.jpg
