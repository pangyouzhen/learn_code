import collections

City = collections.namedtuple('City', 'name couontry population coordinates')

tokyo = City('Tokyo', 'JP', 36.933, (35.68, 139))
print(tokyo)
print(tokyo.population)
print(tokyo[1])
print(City._fields)
Latlong = collections.namedtuple("Latlong", 'lat long')
delhi_data = ('Delhi NCR', "IN", 21, Latlong(28, 77))
delhi = City._make(delhi_data)
print(delhi)
print(delhi._asdict())
# 切片
# 当只有最后一个位置信息时，我们也可以快速看出切片和区间里有几个元素：range(3)和my_list[:3]都返回3个元素。
# 当起止位置信息都可见时，我们可以快速计算出切片和区间的长度，用后一个数减去第一个下标（stop-start）即可。
# 这样做也让我们可以利用任意一个下标来把序列分割成不重叠的两部分，只要写成my_list[:x]和my_list[x:]
