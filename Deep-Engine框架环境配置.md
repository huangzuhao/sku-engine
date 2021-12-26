# Deep-Engine框架环境配置

## 一、显卡驱动安装

由于框架需要相对新的显卡驱动，所以要先确认（nvidia-smi)当前的显卡驱动版本是否为以下：NVIDIA-Linux-x86_64-470.57.02.run

如果不是，需要更新驱动

### 1、卸载旧驱动

可以用以下命令之一卸载驱动，如果命令找不到，则换一个命令试试

```shell
nvidia-uninstall
yum erase kmod-nvidia
rpm -e kmod-nvidia
```

### 2、安装新驱动

```shell
sh NVIDIA-Linux-x86_64-470.57.02.run
```

## 二、安装docker

### 1、安装docker

如果之前没有安装docker，则按以下方式安装

```shell
sudo yum install -y yum-utils
sudo yum-config-manager  --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
```

### 2、设置NVIDIA 容器工具包仓库环境

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
```

### 3、安装nvidia-docker2

```
sudo yum clean expire-cache
sudo yum install -y nvidia-docker2
sudo systemctl restart docker
```

### 4、配置docker存储目录

Docker 默认安装的情况下，会使用 /var/lib/docker/ 目录作为存储目录，用以存放拉取的镜像和创建的容器等。不过由于此目录一般都位于系统盘，遇到系统盘比较小，而镜像和容器多了后就容易尴尬，这里说明一下如何修改 Docker
的存储目录

```shell
vim /etc/docker/daemon.json 
```

增加以下内容并保存，这里保存路径以 /home/docker 为例

```
 "data-root": "/home/docker"
```

改完后的示例如下：

```text
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
     "data-root": "/home/docker"
}

```

改完再重启下，可通过docker info来看是否已更改成功

```text
systemctl restart docker
```

### 5、设置docker开机启动

```
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

## 四、拉取docker镜像

### 1、拉取deep-engine-client

```
docker pull registry.cn-shenzhen.aliyuncs.com/xw_ailab/deep-engine-client:[镜像版本号]
```

### 2、拉取deep-engine-server

```
docker pull registry.cn-shenzhen.aliyuncs.com/xw_ailab/deep-engine-server:[镜像版本号]
```

## 五、启动Deep-Engine服务

### 1、启动deep-engine-client

```
docker run -d --rm --net=host -v/home/WorkSpace/server_codes:/myclient -w /myclient/Deep-Engine -e LANG=C.UTF-8 registry.cn-shenzhen.aliyuncs.com/xw_ailab/deep-engine-client:[镜像版本号]  nohup python Deep-Engine.py --port 9190 &
```

```
-d                         #后台运行 
-v/home/WorkSpace/server_codes:/myclient   #指定共享目录
-w /myclient/Deep-Engine   #指定运行路径
-e LANG=C.UTF-8            #指定环境变量，这里是设置支持中文
```

### 2、启动deep-engine-server

```
docker  run --gpus  all  -d --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/WorkSpace/server_data:/triondata   -e LD_PRELOAD=/mydata/amirstan_plugin/build/lib/libamirstan_plugin.so  -e  CUDA_VISIBLE_DEVICES=0 registry.cn-shenzhen.aliyuncs.com/xw_ailab/deep-engine-server:[镜像版本号]  nohup  tritonserver --model-repository=/triondata/tensorrt_models  --strict-model-config=false --log-verbose=0 &
```

```
-d                                            #后台运行 
-v/home/WorkSpace/server_data:/triondata      #指定共享目录
-e LD_PRELOAD=/mydata/amirstan_plugin/build/lib/libamirstan_plugin.so   #指定tensorrt插件
-e  CUDA_VISIBLE_DEVICES=0                    #指定显卡
```

## 六、docker 基本使用

systemctl start/restart docker 启动docker

docker pull imagename:version 拉取镜像

docker image ls 查看镜像列表

docker ps -a 查看存在的容器

docker ps 查看正在运行的容器

docker restart ID 重启容器

docker stop ID 停止容器

docker exec -it ID bash 进入容器

docker attach ID 进入容器

exit / Ctrl+d 退出并关闭容器

docker tag IMAGEID REPOSITORY:TAG 重命名镜像