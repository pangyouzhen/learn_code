# 更改键盘某些按键（将F1按键转化为w按键）
xmodmap -e 'keycode 67 = w W w W'
xmodmap -e 'keycode 68 = Tab ISO_Left_Tab Tab ISO_Left_Tab'

#docker 相关
# 移除退出的docker
docker ps -a | grep Exit | awk '{print $1}' | xargs docker rm
docker run -d --name elasticsearch  -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:7.3.1
docker run -d -p 9200:9200 -p 5601:5601 nshou/elasticsearch-kibana
# 127.0.0.1:5601

#git
# git 生成ssh
ls ~/.ssh
ssh-keygen
#3. 连续确认三次
cat ~/.ssh/id_rsa.pub
#5. 将得到的内容复制到github上的增加密钥中

#git 删除不用分支
git branch -a | grep -v master | xargs git branch -D
#remove all  remotes
git branch -a | grep -v master >file.log
#使用文件编辑器编辑 移除所有本地的远程分支
git branch -r | grep -v master | xargs git branch -r -D

# systemctl
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
