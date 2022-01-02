# https://github.com/rupa/z 启用z.sh 服务
. ~/z.sh


if [[ $(uname) == "MINGW"* ]]; then
  alias python="winpty python"
#  echo "git bash alias --python-- success"
  # wsl 使用/mnt/d
  # git bash 使用/d/ /c/
  # mobaxterm 使用 /drivers/c
  ln -s /d/ /mnt/d/
  ln -s /c/ /mnt/c/
  #
  ln -s /drivers/c /mnt/c/
  ln -s /drivers/c /mnt/d/
fi

#if [[ $(uname) == "MINGW"* ]]; then
#  . /d/software/miniconda/etc/profile.d/conda.sh
#  echo "conda success"
#fi

alias docker="sudo docker"
alias ll="ls -alh"
#wget 断点续传
alias wget="wget -c"
#递归的创建目录
alias mkdir="mkdir -pv"
alias qv2ray="/home/pang/Downloads/Qv2ray-v2.7.0-linux-x64.AppImage >/dev/null 2>&1"
alias rsync-stock="rsync -avz root@81.71.140.148:/data/project/stock ./ --exclude venv --exclude __pycache__ --exclude .git --exclude .idea --exclude log"
alias s148="ssh root@81.71.140.148"

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
