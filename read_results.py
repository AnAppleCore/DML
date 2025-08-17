#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
from typing import List, Optional, Dict, Any, Tuple

def _find_event_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "events.out.tfevents.*"))
        files = [f for f in files if os.path.isfile(f)]
        return sorted(files, key=os.path.getmtime)
    elif os.path.isfile(path):
        return [path]
    else:
        raise FileNotFoundError(f"Path not found: {path}")

def _read_with_event_accumulator(event_path: str, wanted_tags: Optional[List[str]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    # size_guidance: 0 表示不限制，尽量完整读取
    ea = EventAccumulator(event_path, size_guidance={'scalars': 0})
    ea.Reload()

    available = ea.Tags().get('scalars', [])
    if wanted_tags:
        tags = [t for t in available if t in wanted_tags]
    else:
        tags = available

    rows = []
    for tag in tags:
        for ev in ea.Scalars(tag):
            rows.append({
                "tag": tag,
                "step": ev.step,
                "value": ev.value,
                "wall_time": ev.wall_time,
            })
    return rows, available

def _read_with_tf_summary_iterator(event_path: str, wanted_tags: Optional[List[str]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    # 需要安装 tensorflow
    import tensorflow as tf  # noqa: F401
    from tensorflow.python.summary.summary_iterator import summary_iterator  # type: ignore

    available = set()
    rows = []
    for e in summary_iterator(event_path):
        if not hasattr(e, "summary") or e.summary is None:
            continue
        for v in e.summary.value:
            if v.tag is None:
                continue
            available.add(v.tag)
            if wanted_tags and v.tag not in wanted_tags:
                continue
            # 只读取标量（simple_value）
            if hasattr(v, "simple_value"):
                rows.append({
                    "tag": v.tag,
                    "step": getattr(e, "step", None),
                    "value": v.simple_value,
                    "wall_time": getattr(e, "wall_time", None),
                })
    return rows, sorted(available)

def read_events(event_path: str, tags: Optional[List[str]] = None) -> Tuple[List[Dict[str, Any]], List[str], str]:
    """
    读取单个事件文件，返回 (rows, available_tags, backend)
    rows: [{tag, step, value, wall_time}, ...]
    """
    try:
        rows, available = _read_with_event_accumulator(event_path, tags)
        backend = "tensorboard.EventAccumulator"
        return rows, available, backend
    except ImportError:
        pass  # 尝试 TensorFlow fallback
    except Exception as e:
        # 其他解析错误时，再尝试 TF fallback
        sys.stderr.write(f"[Warn] EventAccumulator failed on {event_path}: {e}\nTrying tensorflow summary_iterator...\n")

    try:
        rows, available = _read_with_tf_summary_iterator(event_path, tags)
        backend = "tensorflow.summary_iterator"
        return rows, available, backend
    except ImportError:
        raise RuntimeError("Neither tensorboard nor tensorflow is available. Please install one of them:\n"
                           "  pip install tensorboard\n"
                           "or\n"
                           "  pip install tensorflow")
    except Exception as e:
        raise RuntimeError(f"Failed to parse event file with tensorflow: {e}")

def main():
    parser = argparse.ArgumentParser(description="Read TensorBoard event file(s) and export scalars.")
    parser.add_argument("path", help="事件文件路径，或包含事件文件的目录")
    parser.add_argument("--tags", type=str, default=None,
                        help="只读取指定 tag，逗号分隔，如: 'train/loss,valid/acc'")
    parser.add_argument("--out", type=str, default=None,
                        help="输出 CSV 路径，如未指定则打印预览")
    parser.add_argument("--latest", action="store_true",
                        help="当 path 为目录时，仅读取最新的一个事件文件")
    parser.add_argument("--list-tags", action="store_true",
                        help="仅列出可用的 scalar tags 不导出")
    args = parser.parse_args()

    tags = [t.strip() for t in args.tags.split(",")] if args.tags else None

    event_files = _find_event_files(args.path)
    if not event_files:
        print("No event files found.")
        sys.exit(1)

    if args.latest and len(event_files) > 1:
        event_files = [event_files[-1]]

    all_rows: List[Dict[str, Any]] = []
    union_tags = set()
    used_backend = None

    for f in event_files:
        rows, available, backend = read_events(f, tags)
        used_backend = used_backend or backend
        union_tags.update(available)
        # 给每行带上文件名，便于多文件区分（可选）
        for r in rows:
            r["file"] = os.path.basename(f)
        all_rows.extend(rows)

    if args.list_tags:
        print("Available scalar tags:")
        for t in sorted(union_tags):
            print(f"  - {t}")
        print(f"(Parsed by {used_backend})")
        return

    if not all_rows:
        print("No scalar data found (check tags or event file).")
        print("Available tags (union):")
        for t in sorted(union_tags):
            print(f"  - {t}")
        sys.exit(0)

    # 尝试用 pandas 导出/展示
    try:
        import pandas as pd  # type: ignore
        import numpy as np  # noqa: F401
        df = pd.DataFrame(all_rows)
        df.sort_values(by=["tag", "step"], inplace=True)
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"Saved {len(df)} rows to: {args.out}")
            print(f"(Parsed by {used_backend})")
        else:
            # 打印每个 tag 的最后 5 行作为预览
            print(f"Parsed by {used_backend}. Preview (last 5 rows per tag):")
            for tag in df["tag"].unique():
                print(f"\n== {tag} ==")
                print(df[df["tag"] == tag].tail(5)[["file", "step", "value", "wall_time"]].to_string(index=False))
    except ImportError:
        # 没有 pandas 就直接打印少量
        print(f"Parsed by {used_backend}. pandas not installed; printing a small preview:")
        per_tag = {}
        for r in all_rows:
            per_tag.setdefault(r["tag"], []).append(r)
        for tag, rows in per_tag.items():
            rows = sorted(rows, key=lambda x: (x.get("step") or 0))
            print(f"\n== {tag} ==")
            for r in rows[-5:]:
                print(f"[{r.get('file','')}] step={r.get('step')} value={r.get('value')} wall_time={r.get('wall_time')}")

if __name__ == "__main__":
    main()