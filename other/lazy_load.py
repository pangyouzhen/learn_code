# https://mp.weixin.qq.com/s?src=11&timestamp=1616655397&ver=2967&signature=7LWfKrb62M56JBPXtFJ8pSGKcfeqoRSrM4m4wGQHfbei53zNXYVHPvm8FMROAY*DjI9fLyBwEsjpbbedrrQzmNGC6On7vyvpUTbP4abomM7upnzRJLJy1PneM-xe4Hcg&new=1
import pymongo


class lazy:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value
