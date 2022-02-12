# https://github.com/rupa/z 启用z.sh 服务
. ~/z.sh


if [[ $(uname) == "MINGW"* ]]; then
  alias python="winpty python"
#  echo "git bash alias --python-- success"
  # wsl,mobaxterm 使用/mnt/d
  # git bash 使用/d/ /c/
  ln -s /d/ /mnt/d/
  ln -s /c/ /mnt/c/
  #
fi

#if [[ $(uname) == "MINGW"* ]]; then
#  . /d/software/miniconda/etc/profile.d/conda.sh
#  echo "conda success"
#fi


lg() {
  #  lazygit
  git pull origin "$(git branch --show-current)"
  # 针对老版本git 没有 -show-current
  #git pull origin "$(git branch | grep '*' | awk '{print $2}')"
  # $? 是显示最后命令的退出状态
  if [ $? -eq 0 ]; then
    echo "-----pull success------- "
    git add .
    git commit -a -m "$1"
    git push origin "$(git branch --show-current)"
    echo "-----push success-------"
  fi
}

# manjaro bash 解压方法
ex() {
  if [ -f $1 ]; then
    case $1 in
    *.tar.bz2) tar xjf $1 ;;
    *.tar.gz) tar xzf $1 ;;
    *.bz2) bunzip2 $1 ;;
    *.rar) unrar x $1 ;;
    *.gz) gunzip $1 ;;
    *.tar) tar xf $1 ;;
    *.tbz2) tar xjf $1 ;;
    *.tgz) tar xzf $1 ;;
    *.zip) unzip $1 ;;
    *.Z) uncompress $1 ;;
    *.7z) 7z x $1 ;;
    *) echo "'$1' cannot be extracted via ex()" ;;
    esac
  else
    echo "'$1' is not a valid file"
  fi
}

 port_path(){ 
  lsof -i:$1 | awk '{print $2}' | grep -v PID | xargs pwdx
 }

alias vv="virtualenv venv"
alias docker_remove_exit="docker ps -a | grep Exit | awk '{print \$1}' | xargs docker rm"
alias docker_restart="sudo systemctl restart docker && sudo chmod 666 /var/run/docker.sock"
alias docker_remove_none="docker images | grep none | awk '{print \$3}' | xargs docker rmi"
alias docker_remove="docker ps -a | grep Exit | awk '{print \$1}' | xargs docker rm && docker images | grep none | awk '{print \$3}' | xargs docker rmi"
alias ll="ls -alh"
#wget 断点续传
alias wget="wget -c"
#递归的创建目录
alias mkdir="mkdir -pv"
alias qv2ray="/home/pang/Downloads/Qv2ray-v2.7.0-linux-x64.AppImage >/dev/null 2>&1"
alias rsync-stock="rsync -avz root@81.71.140.148:/data/project/stock ./ --exclude venv --exclude __pycache__ --exclude .git --exclude .idea --exclude log"
alias s148="ssh root@81.71.140.148"
alias python="python3"

