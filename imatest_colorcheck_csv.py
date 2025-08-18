import os
import numpy as np
from imatest_readcsv_tool import *
from tools import *
class imatest_clorchek_csv:
    def __init__(self):
        self.read = imatest_readcsv_tool()
        self.config = self._init_config()
        self.color_mapping = self._init_color_mapping()

    def _init_config(self):
        return {
            "start_col_letter_c": 'H',
            "end_col_letter_c": 'J',
            "start_col_letter_h": 'B',
            "end_col_letter_H": 'D',
            "start_row": 52,
            "end_row": 75,
            "L-ideal": [38.31, 65.76, 50.12, 44.68, 55.79, 72.44, 62.72, 40.49, 51.53, 31.12, 72.61, 72.6, 28.72, 55.22,
                        42.92, 81.15, 50.64, 50.59, 94.67, 81.95, 67.41, 51.78, 36.14, 20.39],
            "a*-ideal":[14.34, 18.94, -3.59, -12.29, 9.75, -31.22, 35.53, 11.11, 49.69, 23.44, -23.65, 18.36, 15.55,
                         -38.81,50.82, 1.74, 51.3, -30.07, -0.64, -0.95, -0.92, 0.09, -0.57, -0.01],
            "b*-ideal": [14.37, 17.16, -22.68, 23.75, -25.08, -0.99, 55.98, -44.79, 16.34, -20.27, 58.05, 66.44, -50.43,
                         32.43, 28.73, 80.6, -13.41, -28.99, 1.72, 0.16, 0.54, 0.83, -0.37, -1.32],
            "colorblock": ["1(Dark skin)","2(Light skin)","3(Blue sky)","4(Foliage)","5(Blue flower)","6(Bluish green)","7(Orange)","8(Purplish blue)","9(Moderate red)",
                          "10(Purple)","11(Yellow green)","12(Orange yellow)","13(Blue)","14(Green)","15(Red)","16(Yellow)","17(Magenta)","18(Cyan)","19(White)",
                          "20(Neutral 8)","21(Neutral 6.5)","22(Neutral 5)","23(Neutral 3.5)","24(Black)"
                          ],
            "colorblock_index": list(range(1, 25))
        }

    def _init_color_mapping(self):
        return {
            "1(Dark skin)": "#8B4513",  # 深棕色
            "2(Light skin)": "#F5DEB3",  # 浅肤色
            "3(Blue sky)": "#87CEFA",  # 蓝天蓝
            "4(Foliage)": "#32CD32",  # foliage 绿
            "5(Blue flower)": "#9370DB",  # 蓝紫色
            "6(Bluish green)": "#20B2AA",  # 蓝绿色
            "7(Orange)": "#FFA500",  # 橙色
            "8(Purplish blue)": "#483D8B",  # 紫蓝色
            "9(Moderate red)": "#FF69B4",  # 粉红（近似 moderate red）
            "10(Purple)": "#9932CC",  # 紫色
            "11(Yellow green)": "#9ACD32",  # 黄绿
            "12(Orange yellow)": "#FFD700",  # 橙黄
            "13(Blue)": "#0000FF",  # 蓝色
            "14(Green)": "#008000",  # 绿色
            "15(Red)": "#FF0000",  # 红色
            "16(Yellow)": "#FFFF00",  # 黄色
            "17(Magenta)": "#FF00FF",  # 品红
            "18(Cyan)": "#00FFFF",  # 青色
            "19(White)": "#FFFFFF",  # 白色
            "20(Neutral 8)": "#D3D3D3",  # 浅灰色（近似 neutral 8）
            "21(Neutral 6.5)": "#A9A9A9",  # 中灰色（近似 neutral 6.5）
            "22(Neutral 5)": "#808080",  # 深灰色（近似 neutral 5）
            "23(Neutral 3.5)": "#696969",  # 更深灰色（近似 neutral 3.5）
            "24(Black)": "#000000",  # 黑色
            "max":"#FFFFFF",#白色
            "min":"#FFFFFF",
            "mean":"FFFFFF",
            "Saturation":"FFFFFF"
        }

    def lightsource_math(self, txt):
        if "D6" in txt:
            return "D65光源"
        elif "A" in txt:
            return "A光源"
        elif "84" in txt:
            return "TL84光源"
        return None

    def extracted_col(self, data_c, data_h, file):
        ideal = self.config
        df = pd.DataFrame({
            "测试名": file,
            "光源": self.lightsource_math(file),
            "色块": ideal["colorblock"],
            "色块序号": ideal["colorblock_index"],
            "L-ideal": ideal["L-ideal"],
            "a*-ideal": ideal["a*-ideal"],
            "b*-ideal": ideal["b*-ideal"],
        })

        df_c = pd.DataFrame(data_c, columns=["L-meas", "a*-meas", "b*-meas"])
        df_h = pd.DataFrame(data_h, columns=["R-meas", "G-meas", "B-meas"])
        df = pd.concat([df, df_c], axis=1)

        df['Delta E_{ab}'] = np.sqrt((df['L-ideal'] - df['L-meas']) ** 2 +(df['a*-ideal'] - df['a*-meas']) ** 2 +(df['b*-ideal'] - df['b*-meas']) ** 2)
        df['Delta C_{ab}'] = np.sqrt(df['a*-ideal'] ** 2 + df['b*-ideal'] ** 2) -  np.sqrt(df['a*-meas'] ** 2 + df['b*-meas'] ** 2)
        print("meas")
        print(np.sqrt(df['a*-meas'] ** 2 + df['b*-meas'] ** 2).sum() )
        chroma_ratio = (np.sqrt(df['a*-meas'] ** 2 + df['b*-meas'] ** 2).sum() / np.sqrt(df['a*-ideal'] ** 2 + df['b*-ideal'] ** 2).sum())

        df['Color Saturation'] = round(chroma_ratio * 100, 2)

        max_rgb = df_h[["R-meas", "G-meas", "B-meas"]].max(axis=1)
        min_rgb = df_h[["R-meas", "G-meas", "B-meas"]].min(axis=1)
        delta = max_rgb - min_rgb
        df['HSV-Saturation'] = np.where(delta == 0, 0, delta / max_rgb)

        df = df.round(2)
        return df

    def calculate_stats(self, values):
        return [round(max(values), 2), round(min(values), 2), round(np.mean(values), 2)]

    def reprot(self, data):
        light_sources = data['光源'].unique()
        result_e, result_c, result_hsv = [], [], []
        base_blocks = data['色块'].unique().tolist()
        base_blocks += ["max", "min", "mean", "Saturation"]
        hsv_blocks = base_blocks[19:22] + ["max", "min", "mean"]

        result_e.append(base_blocks)
        result_c.append(base_blocks)
        result_hsv.append(hsv_blocks)

        light_list = ["色块"]
        for light in light_sources:
            light_list.append(light)
            col_data = data[data['光源'].str.contains(light)]
            e_vals = col_data['Delta E_{ab}'].tolist()
            c_vals = col_data['Delta C_{ab}'].tolist()
            hsv_vals = col_data['HSV'].tolist()[19:22]
            saturation_vals = col_data['总体饱和度'].tolist()

            result_e.append(e_vals + self.calculate_stats(e_vals) + [f"{self.calculate_stats(saturation_vals)[0]}%"])
            result_c.append(c_vals + self.calculate_stats(c_vals) + [f"{self.calculate_stats(saturation_vals)[0]}%"])
            result_hsv.append(hsv_vals + self.calculate_stats(hsv_vals))
        def to_df(data,light_list):
            df = pd.DataFrame(data).T
            df.columns = light_list
            return df
        return [to_df(result_e,light_list), to_df(result_c,light_list), to_df(result_hsv,light_list)]

    def data_groupby(self, data):
        grouped = data.groupby(['光源', '色块序号', '色块']).agg({
            'Delta E_{ab}': ['count', 'mean'],
            'Delta C_{ab}': ['mean'],
            'Color Saturation': ['mean'],
            'HSV-Saturation': ['mean'],
        }).round(2)

        grouped.columns = ['数量', 'Delta E_{ab}', 'Delta C_{ab}', '总体饱和度', 'HSV']
        return grouped.reset_index()

    def write_csv_format(self, workbook, worksheet, df):
        border_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})
        header_fmt = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1,'bold': True, 'bg_color': '#DDEBF7'})
        for col, val in enumerate(df.columns): worksheet.write(0, col, val, header_fmt)

        color_idx = df.columns.get_loc("色块")
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                worksheet.write(row + 1, col, df.iloc[row, col], border_fmt)
            color_name = df.iloc[row]["色块"]
            worksheet.write(row + 1, color_idx, color_name, workbook.add_format({'bg_color': self.color_mapping.get(color_name, "#FFF"),'align': 'center', 'valign': 'vcenter', 'border': 1}))
        for i, col in enumerate(df.columns):
            width = max(len(str(x)) for x in df[col])
            worksheet.set_column(i, i, min(12, max(width + 2, 10)))

    def write_csv(self, df_all, grouped, report_list, file):
        writer = pd.ExcelWriter(file, engine="xlsxwriter")
        sheets = ["DeltaE报告", "DeltaC报告", "HSV报告", "色彩详细数据", "色彩准确性"]
        dfs = report_list + [df_all, grouped]

        for name, df in zip(sheets, dfs):
            df.to_excel(writer, sheet_name=name, index=False)
            self.write_csv_format(writer.book, writer.sheets[name], df)

        writer.close()

    def read_csv(self, path):
        c_data = self.read.extract_csv_data(self.config["start_col_letter_c"], self.config["end_col_letter_c"],self.config["start_row"], self.config["end_row"], path)
        h_data = self.read.extract_csv_data(self.config["start_col_letter_h"], self.config["end_col_letter_H"], self.config["start_row"], self.config["end_row"], path)
        return self.extracted_col(c_data["data"], h_data["data"], os.path.basename(path))

    def run_all(self, paths, output_file):
        all_data = pd.concat([self.read_csv(p) for p in paths], ignore_index=True)
        grouped = self.data_groupby(all_data)
        report = self.reprot(grouped)
        self.write_csv(all_data, grouped, report, output_file)
if __name__ == '__main__':

    folderPath = r"C:\serialPortVisualization\data\0818_1"

    imagePath,basenames = get_paths(folderPath,suffix=".csv")
    paths = [
        "/home/huangyingwei/WorkSpace/lsc/scripts/ResultsCWF/A_1_summary.csv",
    ]
    output_file = "output.xlsx"
    tool = imatest_clorchek_csv()
    tool.run_all(imagePath, output_file)