1. python 中的垃圾回收机制： 引用计数为主，分代回收为辅

   引用计数为0的时候就会清空该变量，但是整数和短小的字符，Python都会缓存这些对象

   针对循环引用的问题，会进行复制，针对循环的进行-1