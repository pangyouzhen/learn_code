1. python 中的垃圾回收机制： 引用计数为主，分代回收为辅
   
   引用计数为0的时候就会清空该变量，但是整数和短小的字符，Python都会缓存这些对象

   针对循环引用的问题，会进行复制，针对循环的进行-1

1. pycharm 是以当前目录作为工作区的(类似stock目录那样,里面才是stock代码, 入口从main入口进入)
1. pycharm new project + terminal git clone 会造成在工作区多一层的问题