import argparse
import json
import random
import uuid
import os
from typing import List, Dict

def generate_prefab_size() -> int:
    """
    生成单个prefab（层）大小
    使用对数正态分布，平均约27MB，符合真实镜像层分布
    """
    mean_log = 16.5  # 调整后约27MB 对应16.8
    std_log = 1.18    # 标准差
    size = int(random.lognormvariate(mean_log, std_log))
    
    # 限制范围：最小3KB，最大500MB
    size = max(3000, min(size, 500000000))
    return size

def create_shared_prefab_pool(pool_size: int = 1500) -> List[Dict]:
    """
    创建全局共享prefab池
    
    Args:
        pool_size: 共享层池的大小
    
    Returns:
        共享prefab列表
    """
    print(f"创建共享prefab池（{pool_size}个共享层）...")
    shared_prefabs = []
    for _ in range(pool_size):
        prefab = {
            "prefabID": str(uuid.uuid4()),
            "blueprintID": str(uuid.uuid4()),
            "prefabSize": generate_prefab_size()
        }
        shared_prefabs.append(prefab)
    return shared_prefabs

def generate_exclusive_prefabs(num_exclusive: int) -> List[Dict]:
    """
    为镜像生成独占prefabs
    
    Args:
        num_exclusive: 独占层数量
    
    Returns:
        独占prefab列表
    """
    prefabs = []
    for _ in range(num_exclusive):
        prefab = {
            "prefabID": str(uuid.uuid4()),
            "blueprintID": str(uuid.uuid4()),
            "prefabSize": generate_prefab_size()
        }
        prefabs.append(prefab)
    return prefabs

def select_shared_prefabs(shared_pool: List[Dict], num_shared: int) -> List[Dict]:
    """
    从共享池中选择prefabs
    
    Args:
        shared_pool: 共享prefab池
        num_shared: 要选择的数量
    
    Returns:
        选中的共享prefab列表
    """
    if num_shared > 0 and shared_pool:
        return random.sample(shared_pool, min(num_shared, len(shared_pool)))
    return []

def generate_bundle_and_apps(
    num_images: int = 100,
    avg_exclusive_layers: float = 88.8,
    avg_shared_layers: float = 141.4,
    target_avg_size_gb: float = 6.14,
    shared_pool_size: int = 1500
) -> tuple[Dict, Dict]:
    """
    生成Bundle.json和apps.json数据
    
    Args:
        num_images: 镜像数量
        avg_exclusive_layers: 平均独占层数
        avg_shared_layers: 平均共享层数
        target_avg_size_gb: 目标平均大小（GB）
        shared_pool_size: 共享层池大小
    
    Returns:
        (bundle_data, apps_data) 元组
    """
    # 1. 创建全局共享prefab池
    shared_prefab_pool = create_shared_prefab_pool(shared_pool_size)
    
    bundle_data = {}
    apps_data = {}
    
    total_size = 0
    total_layers = 0
    total_exclusive = 0
    total_shared = 0
    
    # print(f"\n开始生成{num_images}个镜像...")
    
    for i in range(1, num_images + 1):
        # 2. 生成该镜像的独占层数量（正态分布）
        num_exclusive = max(70, int(random.gauss(avg_exclusive_layers, 5.0)))
        
        # 3. 生成该镜像的共享层数量（正态分布）
        num_shared = max(120, int(random.gauss(avg_shared_layers, 8.0)))
        
        # 4. 生成镜像名称
        image_name = f"appimg{i}"
        
        # 5. 生成独占prefabs
        exclusive_prefabs = generate_exclusive_prefabs(num_exclusive)
        
        # 6. 选择共享prefabs
        shared_prefabs = select_shared_prefabs(shared_prefab_pool, num_shared)
        
        # 7. 合并所有prefabs
        all_prefabs = exclusive_prefabs + shared_prefabs
        
        # 8. 随机打乱顺序
        random.shuffle(all_prefabs)
        
        # 9. 选择第一个作为taskc（主要层）
        taskc = all_prefabs[0]
        prefabs = all_prefabs[1:]
        
        # 10. 计算镜像总大小
        image_size = sum(p["prefabSize"] for p in all_prefabs)
        
        # 11. 构建Bundle.json数据
        bundle_data[image_name] = {
            "Id": str(uuid.uuid4()),
            "Name": image_name,
            "Tag": "latest",
            "Size": image_size
        }
        
        # 12. 构建apps.json数据
        apps_data[image_name] = {
            "taskc": taskc,
            "prefabs": prefabs
        }
        
        # 统计
        total_size += image_size
        total_layers += len(all_prefabs)
        total_exclusive += num_exclusive
        total_shared += num_shared
        
        # if i % 20 == 0:
        #     print(f"已生成 {i}/{num_images} 个镜像...")
    
    print(f"\n=== 统计信息 ===")
    print(f"镜像总数: {num_images}")
    print(f"平均每镜像层数: {total_layers / num_images:.2f}")
    print(f"平均独占层数: {total_exclusive / num_images:.2f}")
    print(f"平均共享层数: {total_shared / num_images:.2f}")
    print(f"平均镜像大小: {total_size / num_images / (1024**3):.2f} GB")
    print(f"镜像总大小: {total_size / (1024**3):.2f} GB")
    
    return bundle_data, apps_data

def save_json_file(data: Dict, filepath: str, filename: str):
    """
    保存JSON文件
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    name = os.path.join(filepath, filename)
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # print(f"✓ 已保存 {name}")

def analyze_layer_sharing(apps_data: Dict):
    """
    分析层共享情况
    """
    # 统计每个prefabID被多少个镜像使用
    prefab_usage = {}
    
    for image_name, image_data in apps_data.items():
        # taskc
        prefab_id = image_data["taskc"]["prefabID"]
        if prefab_id not in prefab_usage:
            prefab_usage[prefab_id] = []
        prefab_usage[prefab_id].append(image_name)
        
        # prefabs
        for prefab in image_data["prefabs"]:
            prefab_id = prefab["prefabID"]
            if prefab_id not in prefab_usage:
                prefab_usage[prefab_id] = []
            prefab_usage[prefab_id].append(image_name)
    
    # 统计独占和共享
    exclusive_count = sum(1 for imgs in prefab_usage.values() if len(imgs) == 1)
    shared_count = sum(1 for imgs in prefab_usage.values() if len(imgs) > 1)
    
    print(f"\n=== 层共享分析 ===")
    print(f"唯一prefab总数: {len(prefab_usage)}")
    print(f"独占prefab数: {exclusive_count}")
    print(f"可共享prefab数: {shared_count}")
    print(f"平均共享度: {sum(len(imgs) for imgs in prefab_usage.values()) / len(prefab_usage):.2f}")
    print(f"最大共享度: {max(len(imgs) for imgs in prefab_usage.values())}")

def main():
    print("开始生成Taskc容器测试数据...")
    parser = argparse.ArgumentParser()
    parser.add_argument("num", type=int, help="生成的镜像数量", default=100)
    args = parser.parse_args()
    num_images = args.num if args.num > 0 else 100
    
    # 1. 生成数据
    bundle_data, apps_data = generate_bundle_and_apps(
        num_images=num_images,
        avg_exclusive_layers=88.8,
        avg_shared_layers=141.4,
        target_avg_size_gb=6.14,
        shared_pool_size=1500
    )
    
    save_json_file(bundle_data, f"{num_images}-jobs-info/", "Bundles.json")
    save_json_file(apps_data, f"{num_images}-jobs-info/", "apps.json")
    
    analyze_layer_sharing(apps_data)
    
if __name__ == "__main__":
    main()