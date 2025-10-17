import argparse
import json
import random
import uuid
import os
from typing import List, Dict
import hashlib

def generate_sha256_digest() -> str:
    """
    生成随机的sha256 digest
    """
    random_data = str(uuid.uuid4()).encode()
    return "sha256:" + hashlib.sha256(random_data).hexdigest()

def generate_layer_size() -> int:
    """
    生成单层大小，模拟真实Docker镜像层的大小分布
    """
    mean_log = 19.0  # 约300MB的对数均值 对应19.5
    std_log = 1.15    # 标准差
    size = int(random.lognormvariate(mean_log, std_log))
    
    # 限制范围：最小10KB，最大2.5GB
    size = max(10240, min(size, 2500000000))
    return size

def create_layer(digest: str = None) -> Dict:
    """
    创建一个层数据
    
    Args:
        digest: 可选的digest，如果不提供则生成新的
    
    Returns:
        层数据字典
    """
    if digest is None:
        digest = generate_sha256_digest()
    
    return {
        "MIMEType": "application/vnd.oci.image.layer.v1.tar+gzip",
        "Digest": digest,
        "Size": generate_layer_size(),
        "Annotations": None
    }

def create_shared_layer_pool(pool_size: int = 150) -> List[Dict]:
    """
    创建全局共享层池
    
    Args:
        pool_size: 共享层池的大小，默认150个共享层
    
    Returns:
        共享层列表
    """
    shared_layers = []
    for _ in range(pool_size):
        layer = create_layer()
        shared_layers.append(layer)
    return shared_layers

def generate_image_data(
    image_name: str,
    num_exclusive_layers: int,
    num_shared_layers: int,
    shared_layer_pool: List[Dict],
    registry: str = "11.0.1.37:9988/goharbor"
) -> Dict:
    """
    生成单个镜像的完整数据
    
    Args:
        image_name: 镜像名称
        num_exclusive_layers: 独占层数量
        num_shared_layers: 共享层数量
        shared_layer_pool: 全局共享层池
        registry: 镜像仓库地址
    
    Returns:
        镜像数据字典
    """
    # 1. 生成独占层
    exclusive_layers = [create_layer() for _ in range(num_exclusive_layers)]
    
    # 2. 从共享池中随机选择共享层
    shared_layers = []
    if num_shared_layers > 0 and shared_layer_pool:
        shared_layers = random.sample(
            shared_layer_pool,
            min(num_shared_layers, len(shared_layer_pool))
        )
    
    # 3. 合并所有层并随机打乱
    all_layers = exclusive_layers + shared_layers
    random.shuffle(all_layers)
    
    # 4. 提取层的digest列表
    layer_digests = [layer["Digest"] for layer in all_layers]
    
    # 5. 生成镜像数据
    image_data = {
        "Name": f"{registry}/{image_name}",
        "Digest": generate_sha256_digest(),
        "RepoTags": ["latest"],
        "Architecture": "amd64",
        "Os": "linux",
        "Layers": layer_digests,
        "LayersData": all_layers
    }
    
    return image_data

def generate_payload(num_images: int = 100,
    avg_exclusive_layers: float = 20.3,
    avg_shared_layers: float = 1.3,
    target_avg_size_gb: float = 6.14,
    shared_pool_size: int = 150,
    registry: str = "11.0.1.37:9988/goharbor"
) -> Dict:
    """
    生成完整的payload.json数据，支持层共享
    
    Args:
        num_images: 要生成的镜像数量
        avg_exclusive_layers: 平均独占层数
        avg_shared_layers: 平均共享层数
        shared_pool_size: 共享层池大小
    
    Returns:
        包含所有镜像数据的字典
    """
    # 1. 创建全局共享层池
    # print(f"创建共享层池（{shared_pool_size}个共享层）...")
    shared_layer_pool = create_shared_layer_pool(shared_pool_size)
    
    payload = {}
    total_size = 0
    total_layers = 0
    total_exclusive = 0
    total_shared = 0
    
    for i in range(1, num_images + 1):
        #  生成该镜像的独占层数量（正态分布）
        num_exclusive = max(15, int(random.gauss(avg_exclusive_layers, 1.5)))
        
        #  生成该镜像的共享层数量（正态分布）
        num_shared = max(0, int(random.gauss(avg_shared_layers, 0.8)))
        
        image_name = f"testimg{i}"
        image_data = generate_image_data(
            image_name,
            num_exclusive,
            num_shared,
            shared_layer_pool,
            registry
        )
        payload[image_name] = image_data
        
        image_size = sum(layer["Size"] for layer in image_data["LayersData"])
        total_size += image_size
        total_layers += len(image_data["Layers"])
        total_exclusive += num_exclusive
        total_shared += num_shared
    
    print(f"\n=== 统计信息 ===")
    print(f"镜像总数: {num_images}")
    print(f"平均每镜像层数: {total_layers / num_images:.2f}")
    print(f"平均独占层数: {total_exclusive / num_images:.2f}")
    print(f"平均共享层数: {total_shared / num_images:.2f}")
    print(f"平均镜像大小: {total_size / num_images / (1024**3):.2f} GB")
    print(f"镜像总大小: {total_size / (1024**3):.2f} GB")
    return payload

def save_payload_to_file(payload: Dict, filepath: str, filename: str = "payload.json"):
    """
    将生成的payload数据保存到JSON文件
    """
    # 如果目录不存在则创建
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    name = os.path.join(filepath, filename)
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"✓ 已生成 {name}")

def analyze_layer_sharing(payload: Dict):
    """
    分析层共享情况
    """
    # 统计每个层digest被多少个镜像使用
    digest_usage = {}
    
    for image_name, image_data in payload.items():
        for layer_digest in image_data["Layers"]:
            if layer_digest not in digest_usage:
                digest_usage[layer_digest] = []
            digest_usage[layer_digest].append(image_name)
    
    # 统计独占层和共享层
    exclusive_count = sum(1 for imgs in digest_usage.values() if len(imgs) == 1)
    shared_count = sum(1 for imgs in digest_usage.values() if len(imgs) > 1)
    
    print(f"\n=== 层共享分析 ===")
    print(f"唯一层总数: {len(digest_usage)}")
    print(f"独占层数: {exclusive_count}")
    print(f"可共享层数: {shared_count}")
    print(f"平均共享度: {sum(len(imgs) for imgs in digest_usage.values()) / len(digest_usage):.2f}")
    print(f"最大共享度: {max(len(imgs) for imgs in digest_usage.values())}")


def main():
    print("开始生成Docker镜像测试数据...")
    parser  = argparse.ArgumentParser()
    parser.add_argument("num", type=int, help="")
    args = parser.parse_args()
    num_images = args.num
    
    # 生成payload数据
    payload = generate_payload(
        num_images=num_images,
        avg_exclusive_layers=20.3,
        avg_shared_layers=1.3,
        target_avg_size_gb=6.14,
        shared_pool_size=150,
        registry="11.0.1.37:9988/goharbor"
    )
    analyze_layer_sharing(payload)
    save_payload_to_file(payload, f"{num_images}-jobs-info/", "payload.json")

if __name__ == "__main__":
    main()