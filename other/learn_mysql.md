出现过这个问题吗，我们怎么分析的，怎么解决的
1. 慢查询问题 -> 索引
   首先使用mysqldumpslow 进行sql统计，使用explain进行分析
   查询没走索引，
   数据量太大直接走全表
   在索引上进行函数计算
1. 死锁问题
1. 分库分表
1. 业务横向拆分（腾讯技术课堂）
    28定律，top1的写sql占据的时间
1. 存储过程是啥
1. 数据库中的事务是指  几个sql一起执行,或者一起回退
1. 数据库中的两个并行事务，涉及到同一纪录时，定义了四种隔离级别
   Read Uncommitted，读取到了未提交的数据，->脏读
   Read Committed，虽然读取了数据，但是分别在前后读取的，读取的不一致 -> 不可重复读
   Repeatable Read，第一次读的时候发现什么都没有，另一个事务偷偷放了东西进去，再去访问的时候惊讶地居然发现有东西了。 -> 幻读
   如果没有指定隔离级别，数据库就会使用默认的隔离级别。在MySQL中，如果使用InnoDB，默认的隔离级别是Repeatable Read。
   