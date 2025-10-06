
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import sys

def read_log(filepath: str) -> pd.DataFrame:
    """
    读取log文件（逗号分隔格式）
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s*,\s*",
        engine="python",
        skip_blank_lines=True,
        comment="#"
    )
    
    # 统一列名：去除空格和点号，转小写
    df.columns = [c.strip().replace(".", "").replace(" ", "_").lower() for c in df.columns]
    
    # 处理可能的列名变体
    column_mapping = {
        'startabs': ['startabs', 'start_abs'],
        'endabs': ['endabs', 'end_abs', 'edabs', 'ed_abs'],
        'pulledabs': ['pulledabs', 'pulled_abs'],
        'node': ['node'],
        'no': ['no'],
        'ppnum': ['ppnum', 'pp_num']
    }
    
    # 标准化列名
    for standard_name, variants in column_mapping.items():
        for variant in variants:
            if variant in df.columns:
                if standard_name != variant:
                    df = df.rename(columns={variant: standard_name})
                break
    
    # 删除序号列
    if 'no' in df.columns:
        df = df.drop(columns=['no'])
    
    # 期望列存在性检查
    required = {"node", "startabs", "endabs", "pulledabs"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}. 实际列: {list(df.columns)}")

    # 数值列转为数值类型
    num_cols = ["startabs", "endabs", "pulledabs", "jobid", "node", "ppnum"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 去除关键字段缺失的行
    df = df.dropna(subset=["node", "startabs", "endabs", "pulledabs"])

    # 过滤掉非正时长的区间（endabs应该大于startabs）
    valid_duration = df[df["endabs"] > df["startabs"]]
    
    if len(valid_duration) == 0:
        print("警告：没有有效的时长数据！", file=sys.stderr)
        return pd.DataFrame()
    
    df = valid_duration.copy()
    return df

def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    合并重叠区间，返回不相交区间列表
    """
    clean = [(float(s), float(e)) for s, e in intervals if pd.notna(s) and pd.notna(e) and e > s]
    if not clean:
        return []
    clean.sort(key=lambda x: (x[0], x[1]))
    merged = [list(clean[0])]
    for s, e in clean[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def pct(series: pd.Series, q: float) -> float:
    """计算分位数"""
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    return float(s.quantile(q, interpolation="linear"))

def analyze(df: pd.DataFrame, total_nodes: int = 1000):
    """分析集群使用率和多种pulled时间指标"""
    if len(df) == 0:
        return {
            "pulled_p99": float("nan"),
            "pulled_p95": float("nan"),
            "pulled_p90": float("nan"),
            "pulled_p80": float("nan"),
            "pulled_p70": float("nan"),
            "pulled_p60": float("nan"),
            "pulled_p50": float("nan"),
            "pulled_mean": float("nan"),
            "pulled_std": float("nan"),
            "pulled_max": float("nan"),
            "pulled_total": float("nan"),
            "pulled_cv": float("nan"),
            "slow_task_ratio": float("nan"),
            "cluster_utilization": float("nan"),
            "active_nodes": 0,
            "node_effective_utilization": float("nan"),
        }
    
    # 计算pulledtime = pulledAbs - startAbs（镜像拉取时间）
    df["pulledtime"] = df["pulledabs"] - df["startabs"]
    df["pulledtime"] = df["pulledtime"].clip(lower=0)

    # 计算多种pulled时间指标
    pulled_p99 = pct(df["pulledtime"], 0.99)
    pulled_p95 = pct(df["pulledtime"], 0.95)
    pulled_p90 = pct(df["pulledtime"], 0.90)
    pulled_p80 = pct(df["pulledtime"], 0.80)
    pulled_p70 = pct(df["pulledtime"], 0.70)
    pulled_p60 = pct(df["pulledtime"], 0.60)
    pulled_p50 = pct(df["pulledtime"], 0.50)  # 中位数
    pulled_mean = float(df["pulledtime"].mean())
    pulled_std = float(df["pulledtime"].std())
    pulled_max = float(df["pulledtime"].max())
    pulled_total = float(df["pulledtime"].sum())  # 总拉取时间
    
    # 变异系数 (CV = 标准差/均值)
    pulled_cv = pulled_std / pulled_mean if pulled_mean > 0 else float("nan")
    
    # 慢任务比例（拉取时间超过10秒的任务比例）
    slow_tasks = len(df[df["pulledtime"] > 15])
    slow_task_ratio = slow_tasks / len(df) if len(df) > 0 else 0.0

    # 时间窗口（从最早开始到最晚结束）
    min_start_time = float(df["startabs"].min())
    max_end_time = float(df["endabs"].max())
    window_start = min_start_time
    window_end = max_end_time
    window_duration = max(window_end - window_start, 0.0)

    # 每个节点的忙碌时间（基于任务的实际执行时间：startabs到endabs）
    node_busy_time: Dict[str, float] = {}
    for node, g in df.groupby("node"):
        # 使用startabs到endabs作为任务执行区间
        intervals = list(zip(g["startabs"], g["endabs"]))
        clipped = [
            (max(s, window_start), min(e, window_end))
            for s, e in intervals
            if (e > s) and (e > window_start) and (s < window_end)
        ]
        merged = merge_intervals(clipped)
        busy = sum(e - s for s, e in merged)
        node_busy_time[node] = busy

    total_busy_time = float(sum(node_busy_time.values()))
    active_nodes = len(node_busy_time)

    # 时间加权集群使用率 = 总忙碌时间 / (总节点数 × 时间窗口)
    if window_duration > 0 and total_nodes > 0:
        cluster_utilization = total_busy_time / (total_nodes * window_duration)
    else:
        cluster_utilization = float("nan")

    # 节点有效使用率 = 总忙碌时间 / (活跃节点数 × 时间窗口)
    if window_duration > 0 and active_nodes > 0:
        node_effective_utilization = total_busy_time / (active_nodes * window_duration)
    else:
        node_effective_utilization = float("nan")

    results = {
        "pulled_p99": pulled_p99,
        "pulled_p95": pulled_p95,
        "pulled_p90": pulled_p90,
        "pulled_p80": pulled_p80,
        "pulled_p70": pulled_p70,
        "pulled_p60": pulled_p60,
        "pulled_p50": pulled_p50,
        "pulled_mean": pulled_mean,
        "pulled_std": pulled_std,
        "pulled_max": pulled_max,
        "pulled_total": pulled_total,
        "pulled_cv": pulled_cv,
        "slow_task_ratio": slow_task_ratio,
        "cluster_utilization": cluster_utilization,
        "active_nodes": active_nodes,
        "node_effective_utilization": node_effective_utilization,
    }
    return results

def fmt_pct(x: float) -> str:
    """格式化百分比"""
    return f"{x*100:.2f}%" if pd.notna(x) and np.isfinite(x) else "NaN"

def main():
    parser = argparse.ArgumentParser(description="分析集群使用率和pulled时间指标")
    parser.add_argument("filepath", type=str, help="log文件路径")
    parser.add_argument("--total-nodes", type=int, default=1000, help="集群节点总数")
    parser.add_argument("--format", choices=["simple", "detailed"], default="detailed", help="输出格式")
    args = parser.parse_args()

    try:
        df = read_log(args.filepath)
    except Exception as e:
        print(f"读取文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    results = analyze(df, total_nodes=args.total_nodes)
    
    if args.format == "detailed":
        # 详细输出所有指标
        print("=== Pulled Time 指标 ===")
        print(f"P99: {results['pulled_p99']:.2f}s")
        print(f"P95: {results['pulled_p95']:.2f}s")
        print(f"P90: {results['pulled_p90']:.2f}s")
        print(f"P80: {results['pulled_p80']:.2f}s")
        print(f"P70: {results['pulled_p70']:.2f}s")
        print(f"P60: {results['pulled_p60']:.2f}s")
        print(f"P50 (中位数): {results['pulled_p50']:.2f}s")
        print(f"平均值: {results['pulled_mean']:.2f}s")
        print(f"标准差: {results['pulled_std']:.2f}s")
        print(f"最大值: {results['pulled_max']:.2f}s")
        print(f"总拉取时间: {results['pulled_total']:.2f}s")
        print(f"变异系数: {results['pulled_cv']:.3f}")
        print(f"慢任务比例(>15s): {fmt_pct(results['slow_task_ratio'])}")
        print(f"\n=== 集群使用率 ===")
        print(f"时间加权集群使用率: {fmt_pct(results['cluster_utilization'])}")
        print(f"活跃节点数: {results['active_nodes']}")
        print(f"节点有效使用率: {fmt_pct(results['node_effective_utilization'])}")
    else:
        # 简单格式：关键指标
        print(f"{results['pulled_p99']:.2f}, {results['pulled_p95']:.2f}, {results['pulled_mean']:.2f}, {results['pulled_total']:.2f}, {results['pulled_cv']:.3f}, {fmt_pct(results['slow_task_ratio'])}")

if __name__ == "__main__":
    main()
