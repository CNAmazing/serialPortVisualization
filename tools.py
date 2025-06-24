import os
from pathlib import Path

def print_file_tree(directory, indent=''):
    """递归打印目录的文件树结构"""
    path = Path(directory)
    if not path.exists():
        print(f"错误：目录 '{directory}' 不存在")
        return
    
    # 打印当前目录名
    print(f"{indent}📂 {path.name}/")
    
    # 获取所有子项并排序（目录在前，文件在后）
    items = sorted(os.listdir(path))
    dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
    files = [item for item in items if not os.path.isdir(os.path.join(path, item))]
    
    # 递归打印子目录
    for d in dirs:
        print_file_tree(os.path.join(path, d), indent + "    ")
    
    # 打印文件
    for f in files:
        print(f"{indent}    📄 {f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("使用方法: python file_tree.py <目录路径>")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    print(f"文件树: {target_dir}")
    print("-" * 50)
    print_file_tree(target_dir)