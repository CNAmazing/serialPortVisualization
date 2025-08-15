import pandas as pd
import string
import chardet
class   imatest_readcsv_tool:
    def __init__(self):
        pass

    def col_letter_to_index(self,col_letter):
        """将列字母转换为索引（从 0 开始）"""
        index = 0
        for char in col_letter:
            index = index * 26 + (string.ascii_uppercase.index(char) + 1)
        return index - 1

    def extract_csv_data(self,start_col_letter, end_col_letter, start_row, end_row,file_path,sep=','):
        """
        从 CSV 文件中直接提取指定行和列范围的数据，并进行空值判断

        参数:
        start_col_letter (str): 起始列字母
        end_col_letter (str): 结束列字母
        start_row (int): 起始行号
        end_row (int): 结束行号
        file_path (str): CSV 文件路径

        返回:
        dict: 包含提取数据及其空值状态的字典
        """
        start_col_index = self.col_letter_to_index(start_col_letter)
        end_col_index = self.col_letter_to_index(end_col_letter)

        try:
            all_data = []
            all_empty_flags = []
            for row_num in range(start_row, end_row+ 1):
                try:
                    with open(file_path, 'rb') as f:
                        result = chardet.detect(f.read())
                        # 尝试读取当前行的数据
                    df = pd.read_csv(file_path, encoding=result['encoding'], skiprows=row_num - 1, nrows=1,sep=sep)
                    row_data = df.iloc[0, start_col_index:end_col_index + 1].tolist()
                    empty_flags = [pd.isnull(value) for value in row_data]
                    all_data.append(row_data)
                    all_empty_flags.append(empty_flags)
                except pd.errors.ParserError:
                    continue
            result = {
                "data": all_data,
                "is_empty": all_empty_flags
            }
            return result
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到")
        except IndexError:
            print("指定的行或列超出范围")
    def demo(self):
        print("调用测试验证通过")
