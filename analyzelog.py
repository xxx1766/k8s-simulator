
import argparse
import json
import re
from collections import defaultdict

def parse_log_file(file_path):
    """
    解析log文件，提取[SCORE]数据并按要求格式化输出
    """
    # 存储解析后的数据 {job_name: [(score, worker_name), ...]}
    job_data = defaultdict(list)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析JSON
                    log_entry = json.loads(line)
                    log_content = log_entry.get('log', '')
                    
                    # 查找包含[SCORE]的行
                    if '[SCORE]' in log_content:
                        # 修正后的正则表达式，更精确地匹配引号内的内容
                        # 匹配格式: "[SCORE], job-name-no, score, worker-name"
                        pattern = r'\[SCORE\],\s*([^,]+),\s*([^,]+),\s*([^"]+?)(?="|\s*\\r|$)'
                        match = re.search(pattern, log_content)
                        
                        if match:
                            job_name = match.group(1).strip()
                            score = match.group(2).strip()
                            worker_name = match.group(3).strip()
                            
                            try:
                                # 尝试将score转换为数字
                                score_num = float(score)
                                job_data[job_name].append((score_num, worker_name))
                            except ValueError:
                                print(f"警告：第{line_num}行的score无法转换为数字: {score}")
                        else:
                            # 如果第一个正则表达式没有匹配，尝试另一种模式
                            # 直接从引号内提取内容
                            quote_pattern = r'"([^"]*\[SCORE\][^"]*)"'
                            quote_match = re.search(quote_pattern, log_content)
                            if quote_match:
                                quoted_content = quote_match.group(1)
                                # 从引号内容中提取数据
                                data_pattern = r'\[SCORE\],\s*([^,]+),\s*([^,]+),\s*(.+?)(?:\s*$)'
                                data_match = re.search(data_pattern, quoted_content)
                                if data_match:
                                    job_name = data_match.group(1).strip()
                                    score = data_match.group(2).strip()
                                    worker_name = data_match.group(3).strip()
                                    
                                    try:
                                        score_num = float(score)
                                        job_data[job_name].append((score_num, worker_name))
                                    except ValueError:
                                        print(f"警告：第{line_num}行的score无法转换为数字: {score}")
                        
                except json.JSONDecodeError:
                    print(f"警告：第{line_num}行JSON格式错误，跳过")
                    continue
                except Exception as e:
                    print(f"警告：第{line_num}行处理出错: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return
    except Exception as e:
        print(f"错误：读取文件时出错: {e}")
        return
    
    # 处理和输出数据
    if job_data:
        format_and_output(job_data)
    else:
        print("未找到任何[SCORE]数据")

def extract_job_number(job_name):
    """
    从job名称中提取最后一个数字用于排序
    例如: job-2-1 -> 1, job-10-3 -> 3, job-0-10 -> 10
    """
    try:
        # 使用正则表达式提取所有数字，取最后一个
        numbers = re.findall(r'job-\d+-(\d+)', job_name)
        if numbers:
            return int(numbers[-1])  # 返回最后一个数字
    except:
        pass
    return 0  # 如果无法提取数字，返回0

def format_and_output(job_data):
    """
    格式化并输出数据
    """
    # print("清洗后的数据:")
    # print("=" * 50)
    
    # 按job的最后一个数字排序
    sorted_jobs = sorted(job_data.items(), key=lambda x: extract_job_number(x[0]))
    
    for job_name, scores_workers in sorted_jobs:
        # 对每个job的worker按score从大到小排序
        sorted_scores = sorted(scores_workers, key=lambda x: x[0], reverse=True)
        
        # 格式化输出
        worker_scores = []
        for score, worker in sorted_scores:
            worker_scores.append(f"{worker}: {score}")
        
        output_line = f"{job_name}, {', '.join(worker_scores)}"
        # print(output_line)

def export_to_file(job_data, output_file):
    """
    将结果导出到文件
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 按job的最后一个数字排序
            sorted_jobs = sorted(job_data.items(), key=lambda x: extract_job_number(x[0]))
            
            for job_name, scores_workers in sorted_jobs:
                # 对每个job的worker按score从大到小排序
                sorted_scores = sorted(scores_workers, key=lambda x: x[0], reverse=True)
                
                # 格式化输出
                worker_scores = []
                for score, worker in sorted_scores:
                    worker_scores.append(f"{worker}: {score}")
                
                output_line = f"{job_name}, {', '.join(worker_scores)}\n"
                f.write(output_line)
        
        print(f"\n结果已导出到: {output_file}")
    except Exception as e:
        print(f"导出文件时出错: {e}")

def export_top10_scores_simple(job_data, output_file):
    """
    导出每个任务的Top10得分到文件的简化版本
    使用定长格式，每个数据都对齐
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # 按job的最后一个数字排序
            sorted_jobs = sorted(job_data.items(), key=lambda x: extract_job_number(x[0]))
            
            for job_name, scores_workers in sorted_jobs:
                # 对每个job的worker按score从大到小排序，取前10个
                sorted_scores = sorted(scores_workers, key=lambda x: x[0], reverse=True)
                top10_scores = [score for score, _ in sorted_scores[:10]]
                
                # 构建输出行：job名称（定长12字符）+ 每个分数（定长8字符，右对齐）
                line = f"{job_name:<12}"  # 左对齐12字符
                
                for score in top10_scores:
                    if score == int(score):  # 如果是整数，显示为整数
                        line += f"{int(score):>8}"
                    else:  # 如果是小数，保留1位小数
                        line += f"{score:>8.1f}"
                
                # 如果不足10个分数，用空白填充
                for i in range(len(top10_scores), 10):
                    line += f"{'':>8}"
                
                f.write(line + '\n')
        
        # print(f"Top10得分（格式化版）已导出到: {output_file}")
        
    except Exception as e:
        print(f"导出Top10得分时出错: {e}")

# 主程序
if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="")
    parser.add_argument("-o", type=str, help="")
    parser.add_argument("--top10", type=str, help="Top10得分输出文件路径")
    args = parser.parse_args()
    
    input_file = args.f if args.f else "log_file.txt"  # 输入文件路径
    output_file = args.o if args.o else "cleaned_log.txt"  # 输出文件路径
    top10_file = args.top10 if args.top10 else "top10_scores.txt"  # Top10输出文件路径
    
    print("开始处理log文件...")
    
    # 解析log文件
    job_data = defaultdict(list)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析JSON
                    log_entry = json.loads(line)
                    log_content = log_entry.get('log', '')
                    
                    # 查找包含[SCORE]的行
                    if '[SCORE]' in log_content:
                        # 直接从引号内提取内容，这样更准确
                        quote_pattern = r'"([^"]*\[SCORE\][^"]*)"'
                        quote_match = re.search(quote_pattern, log_content)
                        if quote_match:
                            quoted_content = quote_match.group(1)
                            # 从引号内容中提取数据：[SCORE], job-name, score, worker-name
                            parts = quoted_content.split(', ')
                            if len(parts) >= 4:
                                job_name = parts[1].strip()
                                score_str = parts[2].strip()
                                worker_name = parts[3].strip()
                                
                                try:
                                    score_num = float(score_str)
                                    job_data[job_name].append((score_num, worker_name))
                                    # print(f"提取数据: {job_name}, {score_num}, {worker_name}")
                                except ValueError:
                                    print(f"警告：第{line_num}行的score无法转换为数字: {score_str}")
                            else:
                                print(f"警告：第{line_num}行数据格式不正确: {quoted_content}")
                        
                except json.JSONDecodeError:
                    print(f"警告：第{line_num}行JSON格式错误，跳过")
                    continue
                except Exception as e:
                    print(f"警告：第{line_num}行处理出错: {e}")
                    continue
        
        # 格式化并输出数据
        if job_data:
            # print(f"\n总共提取到 {len(job_data)} 个job的数据")
            format_and_output(job_data)
            export_to_file(job_data, output_file)
            export_top10_scores_simple(job_data, top10_file)
        else:
            print("未找到任何[SCORE]数据")
            
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        print("请确保将log文件命名为 'log_file.txt' 并放在同一目录下")
    except Exception as e:
        print(f"错误：读取文件时出错: {e}")
