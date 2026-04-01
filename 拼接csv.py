import pandas as pd
import os
import glob
from typing import List, Optional

def merge_csv_files(
    file_list: List[str],
    output_path: Optional[str] = None,
    check_columns: bool = True,
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    合并多个CSV文件（垂直拼接），要求所有文件具有相同的列结构。
    
    参数:
        file_list: CSV文件路径列表
        output_path: 可选，合并后保存的文件路径（若为None则不保存）
        check_columns: 是否检查所有文件的列名是否一致
        encoding: CSV文件编码，默认utf-8
    返回:
        合并后的DataFrame
    """
    if not file_list:
        raise ValueError("文件列表为空")
    
    dfs = []
    first_columns = None
    
    for i, file in enumerate(file_list):
        df = pd.read_csv(file, encoding=encoding)
        
        if check_columns:
            # 检查列名
            if first_columns is None:
                first_columns = list(df.columns)
            else:
                if list(df.columns) != first_columns:
                    raise ValueError(
                        f"文件 {file} 的列名与第一个文件不一致。\n"
                        f"期望: {first_columns}\n实际: {list(df.columns)}"
                    )
        
        dfs.append(df)
        print(f"已读取: {file} ({len(df)} 行)")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\n合并完成，总行数: {len(merged_df)}")
    
    if output_path:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        merged_df.to_csv(output_path, index=False, encoding=encoding)
        print(f"已保存至: {output_path}")
    
    return merged_df

def merge_from_folder(
    folder_path: str,
    pattern: str = "*.csv",
    output_path: Optional[str] = None,
    check_columns: bool = True,
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    合并指定文件夹下所有匹配模式的CSV文件。
    
    参数:
        folder_path: 文件夹路径
        pattern: 文件名匹配模式，默认 "*.csv"
        output_path: 可选，合并后保存的文件路径
        check_columns: 是否检查所有文件的列名是否一致
        encoding: CSV文件编码
    返回:
        合并后的DataFrame
    """
    file_list = glob.glob(os.path.join(folder_path, pattern))
    if not file_list:
        raise FileNotFoundError(f"未找到匹配 {pattern} 的文件在 {folder_path}")
    return merge_csv_files(file_list, output_path, check_columns, encoding)

if __name__ == "__main__":
    # 示例用法：
    # 1. 通过文件列表手动指定
    files = [
        "data/game1.csv",
        "data/game2.csv",
        "data/browser.csv"
    ]
    # 合并并保存
    merged_df = merge_csv_files(files, output_path="merged_data.csv")
    
    # 2. 合并文件夹下所有csv文件
    # merged_df = merge_from_folder("data/", output_path="all_data.csv")
    
    # 可选：查看合并后的数据概览
    # print(merged_df.head())
