############################
# File Name: download_from_remote.sh
# Author: LPeng
# mail: 7526@qq.com
# Created Time: 2025年04月07日 星期一 12时15分35秒

############################
#!/bin/bash

# # 设置远程主机IP
# remote_host="192.168.100.87"

# # 从命令行参数获取远程路径
# remote_path="$1"

# # 检查是否提供了远程路径参数
# if [ -z "$remote_path" ]; then
#     echo "错误: 请提供要下载的文件或目录路径作为参数。"
#     echo "用法: $0 <远程路径>"
#     exit 1
# fi

# # 使用scp下载文件或目录
# scp -r "root@$remote_host:$remote_path" ./

# # 检查下载是否成功
# if [ $? -eq 0 ]; then
#     echo "下载成功！"
# else
#     echo "下载失败，请检查输入的路径是否正确，以及网络连接是否正常。"
# fi

# ######################################
# File Name: download_from_remote.sh
# Author: LPeng
# mail: 7526@qq.com
# Created Time: 2025年04月07日 星期一 12时15分35秒
# ######################################

# ----------------- 配置区 ----------------- #
# 设置远程主机IP
REMOTE_HOST="192.168.100.87"

# 【修改点 1】设置固定的远程基础目录 (请确保末尾有斜杠 /)
REMOTE_BASE_PATH="/var/ssd_data0/bruce/"
# ------------------------------------------ #


# 从命令行第一个参数获取要下载的文件或目录名
target_name="$1"

# 检查是否提供了参数
if [ -z "$target_name" ]; then
    echo "错误：请提供要下载的文件或目录名作为参数。"
    # 【修改点 2】更新用法提示
    echo "用法: $0 <文件名或目录名>"
    echo "示例 (下载文件): $0 my_model.pth"
    echo "示例 (下载目录): $0 experiment_logs"
    exit 1
fi

# 【修改点 3】将基础目录和目标名拼接成完整的远程路径
remote_path="${REMOTE_BASE_PATH}${target_name}"

echo "准备从远程路径下载: ${remote_path}"

# 使用scp下载文件或目录
# -r 标志允许递归下载目录
scp -r "root@${REMOTE_HOST}:${remote_path}" ./

# 检查下载是否成功
if [ $? -eq 0 ]; then
    echo "下载成功！"
else
    echo "下载失败，请检查远程文件 '${target_name}' 是否存在于 '${REMOTE_BASE_PATH}' 目录下，以及网络连接是否正常。"
fi



#{
#    "graphs": [
#        {
#            "vtcm_mb": 8,
#             "O": 3,
#             "graph_names": [
#                 "model"
#             ]
#         }
#     ],
#     "devices": [
#         {
#             "soc_id": 67,
#             "device_id": 0,
#             "dsp_arch": "v75",
#             "pd_session": "unsigned",
#             "cores": [
#                 {
#                     "core_id": 0,
#                     "perf_profile": "burst"
#                 }
#             ]
#         }
#     ]
# }
