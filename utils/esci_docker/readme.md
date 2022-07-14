构建镜像需要注意的事项
1. 尽量使用公司的源，公司的源一般比较快，性能会比自己机器快，外网的比较全，更加自由
1. 构建镜像前明确python版本，requirements.txt
1. torch的镜像一般使用nvidia/cuda:10.2-devel-ubuntu18.04，torch镜像自带cudnn
1. 镜像中安装python最好使用miniconda，因为conda内部包含的组件更全一些
1. 镜像最后一步，将清空/root/.cache 和 /tmp目录下，减少镜像大小
