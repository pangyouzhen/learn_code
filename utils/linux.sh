# 更改键盘某些按键（将F1按键转化为w按键，F3转化为tab键）
xmodmap -e 'keycode 67 = w W w W'
xmodmap -e 'keycode 69 = Tab ISO_Left_Tab Tab ISO_Left_Tab'
#查看现在键盘的映射
xmodmap -pke

#docker 相关

#ubuntu

docker run -it -d ubuntu:latest --name ubuntu
docker exec -it ubuntu bash
sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
# 一些操作
docker commit d2eb5b9b61f5 nsg_annoy:v1.0
docker save ubuntu:latest >/tmp/annoy_nsg.tar
tar -zcvf nsg-annoy.tgz nsg_annoy.tar
#  这里之所以用 tgz进行压缩，是因为tar只是归档，体积太大影响传输
docker load -i /home/ubuntu/docker/ubuntu.tar
# 移除退出的docker
docker ps -a | grep Exit | awk '{print $1}' | xargs docker rm
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:7.3.1
docker run -d -p 9200:9200 -p 5601:5601 nshou/elasticsearch-kibana
docker run --name mysql -e MYSQL_ROOT_PASSWORD=SeaBiscuit##^ -p 3306:3306 -v /usr/mysql/conf:/etc/mysql/conf.d -v /usr/mysql/data:/var/lib/mysql -d mysql:latest --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci
docker run -it -p 9000:9000 -v /data/faiss:/index docker.io/daangn/faiss-server:latest --help
docker run --name=gridstudio --rm=false -p 8080:8080 -p 4430:4430 docker.io/ricklamers/gridstudio:release
docker run -it -p:4444:4444 retreatguru/headless-chromedriver
# !!!!老版本的docker 运行nvidia
docker run --runtime=nvidia -it -v /data:/data -d tensorflow/tensorflow:1-1.15-gpu
# -v /data:/data 为了挂载数据
docker run -d --name milvus_gpu_0.10.5 --gpus all -p 19530:19530 -p 19121:19121 -v /home/$USER/milvus/db:/var/lib/milvus/db -v /home/$USER/milvus/conf:/var/lib/milvus/conf -v /home/$USER/milvus/logs:/var/lib/milvus/logs -v /home/$USER/milvus/wal:/var/lib/milvus/wal milvusdb/milvus:0.10.5-gpu-d010621-4eda95
docker run -it --network=host -v /path/to/your-project:/tmp/your-project node:8.9 /bin/bash -c 'cd /tmp/your-project && npm install nodejieba --save'
docker build -t stock:v0.1 .

#docker19后新功能
#1，就是docker不需要root权限来启动和运行了
#Got permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Get "http://%2Fvar%2Frun%2Fdocker.sock/v1.24/images/json": dial unix /var/run/docker.sock: connect: permission denied
#解决办法
sudo systemctl restart docker
chmod 666 /var/run/docker.sock
#2，就是支持GPU的增强功能，我们在docker里面想读取nvidia显卡再也不需要额外的安装nvidia-docker了
#docker: Error response from daemon: linux runtime spec devices: could not select device driver "" with capabilities: [[gpu]]
#解决办法
yay -S nvidia-container-runtime

sed -i '/^\[mysqld\]/a default-character-set=utf8mb4' /tmp/my.cnf.bak
sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
echo 'Server = https://mirrors.tuna.tsinghua.edu.cn/manjaro/stable/$repo/$arch' >/etc/pacman.d/mirrorlist

cd /mnt/d/project && echo "success" && rsync -avz ./stock root@81.71.140.148:/tmp  --exclude venv --exclude __pycache__ --exclude .git --exclude .idea  --exclude log --exclude img --exclude raw_data
#git
# git 生成ssh
ls ~/.ssh
ssh-keygen
#3. 连续确认三次
cat ~/.ssh/id_rsa.pub
ssh-copy-id -i id_rsa.pub root@ip
#5. 将得到的内容复制到github上的增加密钥中

#git 删除不用分支
git branch -a | grep -v master | xargs git branch -D
#remove all  remotes
git branch -a | grep -v master >file.log
#使用文件编辑器编辑 移除所有本地的远程分支
git branch -r | grep -v master | xargs git branch -r -D
# git 撤销命令

# git checkout . 撤销所有修改
# git restore --staged .
# git reset HEAD . 用在add之后
# git reset --soft HEAD^ 用在commit之后

# systemctl & systemd
sudo systemctl restart lightdm
sudo systemctl restart NetworkManager
sudo systemctl list-units --type=service --all
#<!-- 分析相关功能耗时 -->
systemd-analyze blame
systemctl list-units --type=service --all | grep "not-found"

# 格式化 json
head -n 1 ./data/dataset/snli_1.0/snli_1.0_train.jsonl | python -m json.tool
echo '{"a": 1, "b": 2}' | python -m json.tool
python -m json.tool djg.txt

#
ssh root@ip

# 查看现在启动着程序的启动参数
cat /proc/pid/cmdline

# 磁盘分区情况
fsck -l

#脚本调试
sh -x xxx.sh

# grep awk sed
# grep 正则  grep的正则使用 -E 参数,使用perl 的正则形式和python正则相差不大
grep -P
#grep -v  取反操作
grep 10.. .xsession-errors
# sh 中 * + ? 重复，这里的+ ?用的时候需要 变成\+ \?
# sed 打印第10行
sed -n "10p" .xsession-errors
sed -n "10,20p" .xsession-errors
# p 是打印的意思
# 使用正则进行匹配 打印
sed -n "/err\+r/p" .xsession-errors
#显示行号
nl .xsession-errors
# 第一个出现 error 和 ERROR 之间的行
sed -n "/error/,/ERROR/p" .xsession-errors
#取反操作
sed -n "10!p" .xsession-errors
sed -n "10,20!p" .xsession-errors
# 每隔两行进行打印
nl .xsession-errors | sed -n "1~2p"

awk '{print $1}' ./src/data/msr_paraphrase_train.txt | sort | uniq -c | sort -n

#关闭plank后面的阴影，
# Window Manager Tweaks - Compositor - Show shadows under dock windows
# 恢复默认xcfe panel
#
#    1. xfce4-panel --quit
#    2. pkill xfconfd
#    First delete settings for the panel, rm -rf ~/.config/xfce4/panel
#    Clear out the settings for xfconfd, rm -rf ~/.config/xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml
#    Restart the panel, run xfce4-panel. This will respawn xfconfd automatically. Note if you need or want to restart xfconfd manually know that on my installation it was in /usr/lib/x86_64-linux-gnu/xfce4/xfconf/xfconfd which was outside of $PATH.

# 更改xcfe的 alt键  tweak
# uml 图, 对模块的继承关系分析时很有用
# pip install pylint
pyreverse --help
pyreverse -ASmn -o png allennlp/data/
pyreverse -o png allennlp/data/ --ignore=a.py,b.py

#时间同步服务
sudo systemctl restart systemd-timesyncd.service
# 将当前时间写入硬件时间
sudo hwclock -w
xrandr --output HDMI-1 --above eDP-1
lspci
inxi -G
unzip -n geekzw-funNLP-master.zip -d ./funNlp

# 将vim的内容复制到系统剪切板 "*y
# 服务器上没界面，不好操作，本地环境不好用 jupyter notebook

#删除软链接
rm 软链接
# 注意这里不能加 /加上就成了删除源文件了
ln -s 源文件 软链接
# 使用命令时 最好先用man 等linux 常用的来看，后面再去百度，一定改掉这个习惯
# 将上一个的输出变成下一个的输入 $ 符号  wc -l $(ls)
# 递增序列
cat -n file.txt >newfile.txt
nl -n ln garch.py >/tmp/garcH_test.py

# 获取输出结果的第一个
#utility 2>&1 | head -n 1

# /dev/null 是空设备，输入的所有东西都将被丢弃

# linux
# !!  上一个命令
# ！$ 上一个命令的参数
# 获取文件的目录
# locate transformer_.py | head -n 1 | xargs dirname
# 搜索以dot结尾的文件
locate dot | grep -P .*?dot$
cd $(locate xfce4-keyboard-shortcuts.xml | head -n 5 | xargs dirname | sed -n '5p')

jupyter lab --allow-root --ip="0.0.0.0" --no-browser >~/jupyter.log 2>&1 &
scp root@ip ./ && echo success >/tmp/scp.log 2>&1 &
# 复制文件 时排除某些文件或者文件夹
rsync -avz --progress ./SimCSE-main /run/media/pang/KINGSTON/ --exclude venv --exclude __pycache__ --exclude .git --exclude .idea

#|       命令      | 标准输出 | 错误输出 | 应用场景 |
#|:---------------:|:--------:|:--------:|----------|
#| >/dev/null 2>&1 | 丢弃     | 丢弃     |程序内有log的|
#| 2>&1 >/dev/null | 丢弃     | 屏幕     |          |
scp root@ip ./ 2>&1 >/dev/null &
# 不要使用scp，使用rsync，rsync -P 是可以断点续传的
rsync -P --rsh='ssh -p 2200' root@81.71.140.148:/data/image2latex_test/image-to-latex/artifacts/model.pt ./
#因为scp的输出不是标准输出 直接>是无效的
sfdp -x -Goverlap=scale -Tpng packages.dot >packages.png

kill -9 pid
#彻底杀死一个进程
kill -STOP pid
#暂停一个进程
kill -CONT pid
#重启一个进程

#找出本文件夹下文件大于2000M的文件并删除
find ./ -type f -size +2000M -exec rm {} \;
#按照文件大小排序
ls -Slh

# 列举定时任务
crontab -l
# 编辑定时任务
crontab -e
#查看crontab 执行情况
tail -100f /var/log/cron
#00 18 * * * /usr/bin/python3 /data/project/stock/main.py

lsof -i:8082 | awk '{print $2}' | grep -v PID | xargs pwdx
cat /proc/pid

# bottle_display
#http://127.0.0.1:8003/visualize.html#mode=edit

#压缩命令/
# 文件传输一定注意要用，可以极大的减少时间
tar -zcvf abc.tgz ./abc

#pigz
tar -cvf ./abc | pigz -p 8 >output.tgz && mv output.tgz /run/meida/pang/K...
pigz -p 8 -d output.tgz

ssh root@81.71.140.148
# linux常见
#执行任务时 & 符号,将任务在后台运行
#如果忘记了&可以ctrl+z 然后bg

# 多个jdk 时
#sudo pacman -S jdk8-openjdk
# ls /usr/lib/jvm/
#从上面的结果中选择一个  sudo archlinux-java set java-12-jdk

sudo npm i jsdom -g

#linux 杀掉自动重启的进程
#https://www.cnblogs.com/Rui6/p/13983713.html

netstat -nltp | grep 8080

# 设置 http 代理
export http=http://127.0.0.1:1089
export https=http://127.0.0.1:1089

# 或, 设置 socket 代理(clash)
export http_proxy=socks5://127.0.0.1:1089
export https_proxy=socks5://127.0.0.1:1089
export all_proxy="https://127.0.0.1:1089"

# 代码阅读
# 带着问题去读！！！!
#  下载相关源码：推荐去 GitHub 上下载，也可以用 Chrome 插件看
#  查看 README.md 和相关说明文档
#  参考 Tutorials 将代码跑起来/找到程序入口
#  利用 Pyreverse 包含在 Pypylint 生成项目框架图 uml图和packages图
#  找到程序入口, 画出时序图
#  找到需要参考的代码，修改
#  机器学习和深度学习重点关注两个点: 数据+模型.数据是怎么处理的,模型的输入和输出是什么

# 问题修复 - 使用 ventoy的 live cd模式
sudo dmesg | grep error
# 进入安全模式
# 修复磁盘
e2fsck -y /dev/sda1

# 构建docker gpu环境
# 预先下载 tensorflow:1.15.5-gpu(默认是3.6) docker +  Miniconda3-py37(可选) + pip.conf + 对应cuda版本的torch(重要)

dot -Tpng classes.dot -o classes.png && viewnior classes.png

# 全局搜索替换
sed -i "s/aaa/AAA/g" $(grep -rl "aaa" ./)

# git 从另一个分支取文件和文件夹
git checkout branch_name -- dirname

conda create -n env_name python=3.7
