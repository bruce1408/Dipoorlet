# import sys
# print(sys.path)
# try:
#     from quant_tools import common_utils
#     print("成功导入 quant_tools")
# except ImportError as e:
#     print(f"导入失败：{e}")


import sys
import os
print(sys.path)
print("\n当前工作目录:", os.getcwd())
print("\n检查 quant_tools 目录是否存在:")
quant_tools_path = "/mnt/share_disk/bruce_trie/Quantizer-Tools/quant_tools"
print(f"目录 {quant_tools_path} 存在: {os.path.exists(quant_tools_path)}")
print(f"目录内容: {os.listdir(quant_tools_path) if os.path.exists(quant_tools_path) else '目录不存在'}")

try:
    from quant_tools import common_utils
    print("成功导入 quant_tools")
except ImportError as e:
    print(f"导入失败：{e}")