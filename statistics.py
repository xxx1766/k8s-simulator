import argparse
import re
from collections import defaultdict

def parse_log_file(log_file_path):
    """
    解析日志文件，统计每个image被调度到的节点
    
    Parameters:
    log_file_path: 日志文件路径
    
    Returns:
    dict: 每个image对应的node列表
    """
    
    image_nodes = defaultdict(set)
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        if not lines:
            print("错误：文件为空")
            return {}
        
        # 跳过第一行（表头）
        header = lines[0].strip()
        print(f"表头: {header}")
        
        # 解析数据行
        for line_num, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            try:
                # 按逗号分割，并去除空格
                parts = [part.strip() for part in line.split(',')]
                
                if len(parts) < 5:  # 确保有足够的列
                    print(f"警告：第{line_num}行数据不完整，跳过: {line}")
                    continue
                
                # 提取需要的字段
                # No, podname, jobid, node, image, startAbs, pulledAbs, edABS, ppnum
                no = parts[0]
                podname = parts[1]
                jobid = parts[2]
                node = parts[3]
                image = parts[4]
                
                # 验证数据有效性
                if image and node:
                    image_nodes[image].add(str(node))
                else:
                    print(f"警告：第{line_num}行的image或node为空，跳过")
                
            except Exception as e:
                print(f"错误：解析第{line_num}行时出错: {e}")
                print(f"问题行: {line}")
                continue
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {log_file_path}")
        return {}
    except Exception as e:
        print(f"错误：读取文件时出错: {e}")
        return {}
    
    # 转换为所需格式并排序
    result = {}
    for image, nodes in image_nodes.items():
        # 将节点按数值排序
        sorted_nodes = sorted(nodes, key=lambda x: int(x) if x.isdigit() else float('inf'))
        result[image] = sorted_nodes
    
    return result

def print_results(image_node_mapping):
    """
    打印结果，格式为 image_name: node_i, node_j,... (X个节点)
    """
    print("\n每个image被调度到的节点统计：")
    print("-" * 60)
    
    # 统计被调度过的node总数
    all_nodes = set()
    for nodes in image_node_mapping.values():
        all_nodes.update(nodes)
    
    total_nodes = len(all_nodes)
    print(f"被调度过的node总数: {total_nodes}")
    print("-" * 60)
    
    # 按image名称排序输出每个image的节点列表和节点数量
    for image in sorted(image_node_mapping.keys()):
        nodes = image_node_mapping[image]
        nodes_str = ", ".join([f"node_{node}" for node in nodes])
        node_count = len(nodes)
        print(f"{image}: {nodes_str} ({node_count}个节点)")
    
    print(f"\n总共有 {len(image_node_mapping)} 个不同的image")

def save_results_to_file(image_node_mapping, output_file="image_node_mapping.txt"):
    """
    将结果保存到文件
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("每个image被调度到的节点统计：\n")
            f.write("-" * 60 + "\n")
            
            # 统计被调度过的node总数
            all_nodes = set()
            for nodes in image_node_mapping.values():
                all_nodes.update(nodes)
            
            total_nodes = len(all_nodes)
            f.write(f"被调度过的node总数: {total_nodes}\n")
            f.write("-" * 60 + "\n")
            
            # 输出每个image的节点列表和节点数量
            for image in sorted(image_node_mapping.keys()):
                nodes = image_node_mapping[image]
                nodes_str = ", ".join([f"node_{node}" for node in nodes])
                node_count = len(nodes)
                f.write(f"{image}, ({node_count}): {nodes_str}\n")
            
            f.write(f"\n总共有 {len(image_node_mapping)} 个不同的image\n")
        
        print(f"结果已保存到 {output_file}")
        
    except Exception as e:
        print(f"保存文件时出错: {e}")

def get_detailed_statistics(image_node_mapping):
    """
    获取详细统计信息
    """
    if not image_node_mapping:
        print("没有数据可统计")
        return
    
    total_images = len(image_node_mapping)
    node_usage = defaultdict(int)
    all_nodes = set()
    
    # 统计每个节点被使用的次数和所有被使用的节点
    for image, nodes in image_node_mapping.items():
        all_nodes.update(nodes)
        for node in nodes:
            node_usage[node] += 1
    
    # 统计每个image被调度到的节点数量
    nodes_per_image = [len(nodes) for nodes in image_node_mapping.values()]
    
    print(f"\n详细统计信息：")
    print(f"- 总image数量: {total_images}")
    print(f"- 被调度过的node总数: {len(all_nodes)}")
    print(f"- 平均每个image被调度到的节点数: {sum(nodes_per_image) / len(nodes_per_image):.2f}")
    print(f"- 最多被调度到的节点数: {max(nodes_per_image)}")
    print(f"- 最少被调度到的节点数: {min(nodes_per_image)}")
    
    # 统计节点数量分布
    node_count_distribution = defaultdict(int)
    for count in nodes_per_image:
        node_count_distribution[count] += 1
    
    print(f"\n节点数量分布：")
    for node_count in sorted(node_count_distribution.keys()):
        image_count = node_count_distribution[node_count]
        print(f"  {node_count}个节点: {image_count}个image")
    
    # 显示所有被使用的节点（按数值排序）
    sorted_all_nodes = sorted(all_nodes, key=lambda x: int(x) if x.isdigit() else float('inf'))
    print(f"\n所有被使用的节点: {', '.join([f'node_{node}' for node in sorted_all_nodes])}")
    
    # 显示最常用的节点
    sorted_nodes = sorted(node_usage.items(), key=lambda x: x[1], reverse=True)
    print(f"\n节点使用频率排行：")
    for i, (node, count) in enumerate(sorted_nodes, 1):
        print(f"  {i}. node_{node}: {count}次")
        if i >= 10:  # 只显示前10名
            break

def show_node_count_summary(image_node_mapping):
    """
    显示每个image的节点数量汇总
    """
    print(f"\n每个image的节点数量汇总：")
    print("-" * 40)
    
    for image in sorted(image_node_mapping.keys()):
        node_count = len(image_node_mapping[image])
        print(f"{image}: {node_count}个节点")

def validate_data(image_node_mapping, expected_images=100):
    """
    验证数据完整性
    """
    actual_images = len(image_node_mapping)
    
    # 统计总的节点数
    all_nodes = set()
    for nodes in image_node_mapping.values():
        all_nodes.update(nodes)
    
    print(f"\n数据验证：")
    print(f"- 期望的image数量: {expected_images}")
    print(f"- 实际的image数量: {actual_images}")
    print(f"- 被调度过的node总数: {len(all_nodes)}")
    
    if actual_images == expected_images:
        print("✓ 数据完整")
    elif actual_images < expected_images:
        print(f"⚠ 缺少 {expected_images - actual_images} 个image")
    else:
        print(f"⚠ 多出 {actual_images - expected_images} 个image")

# 主程序
if __name__ == "__main__":
    
    parser  = argparse.ArgumentParser()
    parser.add_argument("-f", type=str, help="")
    parser.add_argument("-o", type=str, help="")
    args = parser.parse_args()
    # 解析日志文件
    image_mapping = parse_log_file(args.f)
    
    if image_mapping:
        # 打印结果
        # print_results(image_mapping)
        
        # 保存到文件
        save_results_to_file(image_mapping, output_file=args.o)
        
        # 显示节点数量汇总
        # show_node_count_summary(image_mapping)
        
        # # 显示统计信息
        # get_detailed_statistics(image_mapping)
        
        # # 验证数据（假设有100个image）
        # validate_data(image_mapping, 100)
        
    else:
        print("未能解析到任何数据")

# # 额外功能：查找特定image的调度情况
# def find_image_scheduling(image_node_mapping, image_name):
#     """
#     查找特定image的调度情况
#     """
#     if image_name in image_node_mapping:
#         nodes = image_node_mapping[image_name]
#         nodes_str = ", ".join([f"node_{node}" for node in nodes])
#         print(f"\n{image_name} 被调度到: {nodes_str}")
#         print(f"总共调度到 {len(nodes)} 个节点")
#     else:
#         print(f"\n未找到 {image_name} 的调度记录")

# # 额外功能：查找特定节点的image列表
# def find_node_images(image_node_mapping, target_node):
#     """
#     查找特定节点上运行的所有image
#     """
#     images_on_node = []
#     for image, nodes in image_node_mapping.items():
#         if str(target_node) in nodes:
#             images_on_node.append(image)
    
#     if images_on_node:
#         print(f"\nnode_{target_node} 上运行的image:")
#         for img in sorted(images_on_node):
#             print(f"  - {img}")
#         print(f"总共 {len(images_on_node)} 个image")
#     else:
#         print(f"\nnode_{target_node} 上没有运行任何image")

# 使用示例
"""
# 查找特定image
find_image_scheduling(image_mapping, "testimg10")

# 查找特定节点
find_node_images(image_mapping, "45")
"""