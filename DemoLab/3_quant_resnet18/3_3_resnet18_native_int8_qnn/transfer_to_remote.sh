##########################
# File Name: transfer_to_remote.sh
# Author bruce 
# mail: cuidongdong@haomo.ai
# Created Time: 2024年07月17日 星期三 13时15分55秒
############################
#!/bin/bash
############################                                                                                                                                                              
# 使用scp传输tar文件
TAR_FILE=$1


info_error() {
    local message=$1
    local progress=$2
    local color='\033[1;31m'
    echo -e "$color$message\033[0m"
}


# 检查输入参数
if [ "$#" -ne 1 ]; then
    info_error "ERROR: $0 target_files"
    exit 1
fi

echo "======== the md5 value is "
md5sum ${TAR_FILE}
DESTINATION_PATH=/var/ssd_data0/bruce
scp -r "$TAR_FILE" root@192.168.100.87:${DESTINATION_PATH}
