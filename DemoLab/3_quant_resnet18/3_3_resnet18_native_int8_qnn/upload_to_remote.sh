#!/bin/bash

# 设置远程主机信息
remote_host="678e42c621e1f0492395f6e8-1737376454@ve-ssh-proxy.haomoai.com"
remote_port="11122"
remote_dir="/mnt/share_disk/bruce_trie/outputs"

# 从命令行参数获取本地文件或目录路径
local_path="$1"

# 检查是否提供了本地路径参数
if [ -z "$local_path" ]; then
    echo "错误: 请提供要上传的本地文件或目录路径作为参数。"
    echo "用法: $0 <本地路径>"
    exit 1
fi



echo "the passwd is: VRMqnsArUPa1TzY"

# 使用scp上传文件或目录，添加额外的SSH选项
scp -P $remote_port -r -o "StrictHostKeyChecking=no" -o "HostKeyAlgorithms=+ssh-rsa" -o "PubkeyAcceptedKeyTypes=+ssh-rsa" "$local_path" "$remote_host:$remote_dir"

# 检查上传是否成功
if [ $? -eq 0 ]; then
    echo "上传成功！文件或目录已上传到 $remote_host:$remote_dir"
else
    echo "上传失败，请检查输入的路径是否正确，以及网络连接是否正常。"
    echo "错误详情："
    scp -P $remote_port -r -o "StrictHostKeyChecking=no" -o "HostKeyAlgorithms=+ssh-rsa" -o "PubkeyAcceptedKeyTypes=+ssh-rsa" -v "$local_path" "$remote_host:$remote_dir"
fi
