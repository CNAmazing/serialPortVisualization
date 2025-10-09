import os
from pathlib import Path

def write_file_tree(directory, output_file, indent=''):
    """
    递归将目录的文件树结构写入文本文件
    :param directory: 要遍历的目录路径
    :param output_file: 输出文件路径
    :param indent: 缩进字符串，用于格式化
    """
    path = Path(directory)
    
    # 检查目录是否存在
    if not path.exists():
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"错误：目录 '{directory}' 不存在\n")
        return
    
    # 写入当前目录名
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"{indent}📂 {path.name}/\n")
    
    # 获取所有子项并排序（目录在前，文件在后）
    try:
        items = sorted(os.listdir(path))
        dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
        files = [item for item in items if not os.path.isdir(os.path.join(path, item))]
        
        # 递归处理子目录
        for d in dirs:
            write_file_tree(os.path.join(path, d), output_file, indent + "    ")
        
        # 写入文件
        with open(output_file, 'a', encoding='utf-8') as f:
            for file in files:
                f.write(f"{indent}    📄 {file}\n")
    except PermissionError:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"{indent}    🔒 无权限访问此目录\n")

# 使用示例
if __name__ == "__main__":
    target_directory = r"C:\WorkSpace\libcamera\src"
    output_filename = "file_tree.txt"
    
    # 清空或创建输出文件
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"文件树结构: {target_directory}\n\n")
    
    write_file_tree(target_directory, output_filename)
    print(f"文件树已成功写入到 {output_filename}")