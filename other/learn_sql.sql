--java -jar eureka-0.0.1-SNAPSHOT.jar
-- docker run -it --rm -p 3306:3306  -e MYSQL_ROOT_PASSWORD=pw3306 -d  mysql:latest
-- 2. docker exec -it 容器ID  bash
-- 稍等，等mysql 创建完 mysqld.sock，不然会出现Can't connect to local MySQL server through socket '/var/run/mysqld/mysqld.sock' (2)
-- 3. mysql -u root -p
-- 创建表 create database springcloud_sell;

create table product_category
(
  `category_id`   int         not null auto_increment,
  `category_name` varchar(64) not null,
  `category_type` int         not null,
  `create_time`   timestamp   not null default current_timestamp,
  `update_time`   timestamp   not null default current_timestamp on update current_timestamp,
  primary key (`category_id`),
  unique key `uqe_category_type` (`category_type`)

);

insert into product_category (category_id, category_name, category_type, create_time, update_time)
values (1, '热榜', 11, '2017-03-28 16:40:22', '2017-11-26 23:29;36'),
       (2, '好吃的', 22, '2017-03-14 17:38:46', '2017-11-26 23:39:40');

create table `product_info`
(
  `product_id`     varchar(32)   not null,
  `product_name`   varchar(64)   not null,
  `product_price`  decimal(8, 2) not null,
  `product_stock`  int           not null,
  `product_desc`   varchar(64),
  `product_icon`   varchar(512),
  `product_status` tinyint(3)             default 0,
  `category_type`  int           not null,
  `create_time`    timestamp     not null default current_timestamp,
  `update_time`    timestamp     not null default current_timestamp on update current_timestamp,
  primary key (`product_id`)
);

insert into product_info (product_id, product_name, product_price, product_stock, product_desc, product_icon, product_status, category_type) values ('111','皮蛋粥',0.01,39,'','',0,1),('222','蛋糕',10,200,'','',1,1);


create table order_master (
  order_id varchar(32) not null,
  buyer_name varchar(32) not null,
  buyer_phone varchar(32) not null ,
  buyer_address varchar(128) not null ,
  buyer_openid varchar(64) not null ,
  order_amount decimal(8,2) not null ,
  order_status int(3)  not null ,
  pay_status int(3) not null ,
  `create_time`    timestamp     not null default current_timestamp,
  `update_time`    timestamp     not null default current_timestamp on update current_timestamp,
  primary key (order_id),
  key idx_buyer_openid (buyer_openid)
);

drop table order_master;

create table order_detail(
  `detail_id` varchar(32) not null ,
  `order_id` varchar(32) not null ,
  `product_id` varchar(32) not null ,
  `product_name` varchar(64) not null ,
  `product_price` decimal(8,2) not null ,
  `product_quantity` int(3) not null ,
  `product_icon` varchar(512),
  `create_time`    timestamp     not null default current_timestamp,
  `update_time`    timestamp     not null default current_timestamp on update current_timestamp,
  primary key (detail_id),
  key idx_order_id (order_id),
  foreign key (order_id) references order_master(order_id)

);

truncate table order_detail;
truncate table order_master;
delete from order_master;


insert into order_master(order_id) VALUES
("158294512063778197");