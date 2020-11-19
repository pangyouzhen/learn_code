# 更改键盘某些按键（将F1按键转化为w按键，F3转化为tab键）
xmodmap -e 'keycode 67 = w W w W'
xmodmap -e 'keycode 69 = Tab ISO_Left_Tab Tab ISO_Left_Tab'
#查看现在键盘的映射
xmodmap -pke

#docker 相关
# 移除退出的docker
docker ps -a | grep Exit | awk '{print $1}' | xargs docker rm
docker run -d --name elasticsearch  -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:7.3.1
docker run -d -p 9200:9200 -p 5601:5601 nshou/elasticsearch-kibana
docker run -it --name mysql --rm -p 3306:3306  -e MYSQL_ROOT_PASSWORD=password -d  mysql:latest
# 127.0.0.1:5601

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
git branch -a | grep -v master > file.log
#使用文件编辑器编辑 移除所有本地的远程分支
git branch -r | grep -v master | xargs git branch -r -D
# docker 打包镜像
# docker load 镜像

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
# grep 正则
grep 10.. .xsession-errors
# sh 中 * + ? 重复，这里的+ ?用的时候需要 变成\+ \?
# () 在 grep 正则中也要 \( \)
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
# uml 图 + 时序图
# pip install pylint
#pyreverse -ASmy -o png allennlp/data/

# pycharm struct 说明
# c class
# v 类变量
# p 方法变属性
# m 方法
# f 属性