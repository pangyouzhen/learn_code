#更改键盘某些按键（将F1按键转化为w按键）
xmodmap -e 'keycode 67 = w W w W'

#移除退出的docker
docker ps -a | grep Exit | awk '{print $1}' | xargs docker rm


#git 生成ssh
#1. 查看ls ~/.ssh
#2. ssh-keygen
#3. 连续确认三次
#4. cat ~/.ssh/id_rsa.pub
#5. 将得到的内容复制到github上的增加密钥中
