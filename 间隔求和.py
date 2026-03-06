import pandas as pd

def sum_columns_by_interval(csv_filename, column_names, interval_rows, start_from_zero=True):
    """
    对CSV文件中每间隔x行的某些列进行求和
    
    参数:
    csv_filename: str, CSV文件名
    column_names: list, 需要操作的列名列表
    interval_rows: int, 间隔行数
    start_from_zero: bool, 是否从第0行开始（True=从第0行，False=从第1行）
    
    返回:
    list: 每间隔x行的指定列求和结果列表
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_filename)
        
        # 验证列名是否存在
        missing_columns = [col for col in column_names if col not in df.columns]
        if missing_columns:
            raise ValueError(f"列名不存在: {missing_columns}")
        
        # 初始化结果列表
        result = []
        
        # 确定起始行
        start_row = 0 if start_from_zero else 1
        
        # 遍历每隔interval_rows的行
        for i in range(start_row, len(df), interval_rows):
            # 获取当前行指定列的值并求和
            row_sum = df.loc[i, column_names].sum()
            result.append(row_sum)
        
        return result
    
    except FileNotFoundError:
        print(f"错误: 文件 '{csv_filename}' 不存在")
        return []
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return []

# 测试第二个版本
if __name__ == "__main__":
    # 创建测试数据
    test_data = {
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }
    
    test_df = pd.DataFrame(test_data)
    
    print("原始数据:")
    print(test_df)
    
    # 从第0行开始
    result1 = sum_columns_by_interval('test.csv', ['A', 'B'], 3, start_from_zero=True)
    print(f"\n从第0行开始，每间隔3行求和: {result1}")
    
    # 从第1行开始
    result2 = sum_columns_by_interval('test.csv', ['A', 'B'], 3, start_from_zero=False)
    print(f"从第1行开始，每间隔3行求和: {result2}")
