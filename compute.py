#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import sys

def read_log(filepath: str) -> pd.DataFrame:
    """
    读取逗号分隔的日志文件（带表头），对字段做清洗、类型转换。
    支持形如：No., job, node, running_pods, score, compute_ready_at, data_ready_at, start_time, end_time
    """
    # 使用正则分隔，允许逗号两侧有空格
    df = pd.read_csv(
        filepath,
        sep=r"\s*,\s*",
        engine="python",
        skip_blank_lines=True,
        comment="#"
    )
    # 统一列名：小写、去点、空格转下划线
    df.columns = [c.strip().lower().replace(".", "").replace(" ", "_") for c in df.columns]

    # 期望列存在性检查（允许多余列）
    required = {"node", "start_time", "end_time", "compute_ready_at", "data_ready_at"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}. 实际列: {list(df.columns)}")

    # 数值列转为数值类型
    num_cols = ["running_pods", "score", "compute_ready_at", "data_ready_at", "start_time", "end_time"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 去除关键字段缺失的行
    df = df.dropna(subset=["node", "start_time", "end_time", "compute_ready_at", "data_ready_at"])

    # 过滤掉非正时长的区间（如果有 end_time <= start_time 的异常数据）
    df = df[df["end_time"] > df["start_time"]].copy()

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
    s = series.dropna()
    if len(s) == 0:
        return float("nan")
    return float(s.quantile(q, interpolation="linear"))

def analyze(df: pd.DataFrame, total_nodes: int = 1000, window_mode: str = "min_to_max"):
    # 计算等待时长（下截断为 0，避免负值进入统计）
    df["compute_wait"] = (df["compute_ready_at"] - df["start_time"]).clip(lower=0)
    df["data_wait"] = (df["data_ready_at"] - df["start_time"]).clip(lower=0)

    compute_total = float(df["compute_wait"].sum())
    compute_p99 = pct(df["compute_wait"], 0.99)
    compute_p95 = pct(df["compute_wait"], 0.95)

    data_total = float(df["data_wait"].sum())
    data_p99 = pct(df["data_wait"], 0.99)
    data_p95 = pct(df["data_wait"], 0.95)

    max_end_time = float(df["end_time"].max())
    min_start_time = float(df["start_time"].min())

    if window_mode == "min_to_max":
        window_start = min_start_time
    elif window_mode == "zero_to_max":
        window_start = 0.0
    else:
        raise ValueError("window_mode 必须是 'min_to_max' 或 'zero_to_max'")

    window_end = max_end_time
    window_duration = max(window_end - window_start, 0.0)

    # 每个节点的忙时（区间裁剪到观测窗口，再做并集）
    node_busy_time: Dict[str, float] = {}
    for node, g in df.groupby("node"):
        intervals = list(zip(g["start_time"], g["end_time"]))
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

    if window_duration > 0 and total_nodes > 0:
        cluster_utilization = total_busy_time / (total_nodes * window_duration)
    else:
        cluster_utilization = float("nan")

    active_node_ratio = (active_nodes / total_nodes) if total_nodes > 0 else float("nan")

    per_node_util = {
        n: (t / window_duration if window_duration > 0 else float("nan"))
        for n, t in node_busy_time.items()
    }

    results = {
        "compute_total": compute_total,
        "compute_p99": compute_p99,
        "compute_p95": compute_p95,
        "data_total": data_total,
        "data_p99": data_p99,
        "data_p95": data_p95,
        "max_end_time": max_end_time,
        "window_start": window_start,
        "window_end": window_end,
        "window_duration": window_duration,
        "cluster_utilization": cluster_utilization,
        "active_nodes": active_nodes,
        "active_node_ratio": active_node_ratio,
        "total_nodes": total_nodes,
        "per_node_util": per_node_util,  # 如需导出可用
    }
    return results

def fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%" if pd.notna(x) and np.isfinite(x) else "NaN"

def main():
    parser = argparse.ArgumentParser(description="Analyze node utilization and waits from log file")
    parser.add_argument("filepath", type=str, help="日志文件路径（逗号分隔，带表头）")
    parser.add_argument("--total-nodes", type=int, default=1000, help="集群节点总数，默认 1000")
    parser.add_argument("--window", type=str, choices=["min_to_max", "zero_to_max"], default="min_to_max",
                        help="观测窗口口径，默认 min_to_max")
    parser.add_argument("--export-per-node", type=str, default=None,
                        help="可选：导出每个出现过节点的使用率到 CSV 文件（列：node, utilization）")
    args = parser.parse_args()

    try:
        df = read_log(args.filepath)
    except Exception as e:
        print(f"读取日志失败: {e}", file=sys.stderr)
        sys.exit(1)

    results = analyze(df, total_nodes=args.total_nodes, window_mode=args.window)
    # print("compute_wait_time, compute_wait_P99, compute_wait_P95, data_wait_time, data_wait_P99, data_wait_P95, 时间加权集群使用率, 活跃节点占比, 出现过的节点数/总节点数, max_end_time")
    # print(
    #     f"{results['compute_total']:.2f}, "
    #     f"{results['compute_p99']:.2f}, "
    #     f"{results['compute_p95']:.2f}, "
    #     f"{results['data_total']:.2f}, "
    #     f"{results['data_p99']:.2f}, "
    #     f"{results['data_p95']:.2f}, "
    #     f"{fmt_pct(results['cluster_utilization'])}, "
    #     f"{fmt_pct(results['active_node_ratio'])}, "
    #     f"{results['active_nodes']}/{results['total_nodes']}, "
    #     f"{results['max_end_time']:.2f}")
    print(f"{fmt_pct(results['cluster_utilization']*1000/results['active_nodes'])}")

    if args.export_per_node:
        per_node_series = pd.Series(results["per_node_util"], name="utilization").sort_values(ascending=False)
        out_df = per_node_series.reset_index().rename(columns={"index": "node"})
        out_df.to_csv(args.export_per_node, index=False)
        # print(f"\n已导出每个节点使用率到: {args.export_per_node}")

if __name__ == "__main__":
    main()