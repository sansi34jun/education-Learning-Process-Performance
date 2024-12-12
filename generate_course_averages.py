import os
import pandas as pd
import numpy as np

def process_files(directory):
    # 存储所有学生的数据
    student_data = {}
    
    # 预定义的课程性质
    predefined_course_types = [
        '专业必修课',
        '学科基础必修课',
        '通识必修课',
        '通识公共选修课'
    ]
    
    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx') or filename.endswith('.xls'):
            file_path = os.path.join(directory, filename)
            try:
                # 读取Excel文件，不使用第一行作为列名
                df = pd.read_excel(file_path, header=0)
                
                print(f"处理文件: {filename}")
                print("列名:", df.columns.tolist())  # 打印列名以便调试
                
                # 确保必要的列存在（不区分大小写）
                df.columns = [str(col).strip() for col in df.columns]  # 清理列名
                name_col = next((col for col in df.columns if '姓名' in str(col)), None)
                course_type_col = next((col for col in df.columns if '课程性质' in str(col)), None)
                score_col = next((col for col in df.columns if '成绩' in str(col)), None)
                
                if not all([name_col, course_type_col, score_col]):
                    print(f"警告: {filename} 缺少必要的列")
                    continue
                
                # 处理每个学生的数据
                for name in df[name_col].unique():
                    if pd.isna(name):  # 跳过空值
                        continue
                    
                    student_df = df[df[name_col] == name]
                    
                    # 按课程性质分组计算平均分
                    course_averages = student_df.groupby(course_type_col)[score_col].mean()
                    
                    # 如果学生已存在，更新数据；否则创建新条目
                    if name in student_data:
                        for course_type, avg_score in course_averages.items():
                            if course_type in student_data[name]:
                                # 如果已存在该课程性质，取平均值
                                student_data[name][course_type] = np.mean([
                                    student_data[name][course_type], 
                                    avg_score
                                ])
                            else:
                                student_data[name][course_type] = avg_score
                    else:
                        student_data[name] = course_averages.to_dict()
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                import traceback
                print(traceback.format_exc())  # 打印详细错误信息
    
    # 创建最终的DataFrame
    if not student_data:
        print("警告: 没有找到任何学生数据")
        return pd.DataFrame(columns=['姓名'] + predefined_course_types)
    
    final_df = pd.DataFrame.from_dict(student_data, orient='index')
    
    # 确保所有预定义的课程性质列都存在
    for course_type in predefined_course_types:
        if course_type not in final_df.columns:
            final_df[course_type] = np.nan
    
    # 只保留预定义的课程性质列
    final_df = final_df.reindex(columns=predefined_course_types)
    
    # 重置索引，将姓名作为一列
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'index': '姓名'}, inplace=True)
    
    # 格式化成绩为两位小数
    for col in final_df.columns:
        if col != '姓名':
            final_df[col] = final_df[col].round(2)
    
    return final_df

def main():
    # 指定目录路径
    directory = '/Users/crj/learn1/2'
    
    # 处理文件并获取结果
    result_df = process_files(directory)
    
    # 保存结果到Excel文件
    output_path = 'course_type_averages.xlsx'
    result_df.to_excel(output_path, index=False)
    
    print(f"\n处理完成，结果已保存到 {output_path}")
    print("\n数据预览:")
    print(result_df.head())
    
    # 打印一些统计信息
    print(f"\n总共处理了 {len(result_df)} 名学生的数据")
    print("\n各类课程的统计信息:")
    print(result_df.describe())

if __name__ == "__main__":
    main() 