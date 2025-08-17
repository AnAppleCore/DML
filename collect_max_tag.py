#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect the maximum value of a given scalar tag (default: "valid_acc_1")
from each subdirectory under a TensorBoard logs root, and write a CSV.

Usage examples:
  python DML/collect_max_tag.py DML/logs --tag valid_acc_1 --out DML/logs/max_valid_acc_1.csv
  python DML/collect_max_tag.py /home/yanhongwei/spaced_kd/online_kd/DML/logs --tag valid_acc_1
"""

import os
import argparse
from typing import Dict, Any, List, Optional

# Reuse the existing event reading utilities
from read_results import _find_event_files, read_events  # type: ignore


def compute_max_for_run(run_dir: str, tag: str, latest_only: bool = False) -> Optional[Dict[str, Any]]:
    """Compute max value of `tag` across all event files in run_dir.

    Returns a dict {run, max_value, max_step, max_file} or None if no data.
    """
    event_files = _find_event_files(run_dir)
    if not event_files:
        return None
    if latest_only and len(event_files) > 1:
        event_files = [event_files[-1]]

    max_value = None
    max_step = None
    max_file = None

    for f in event_files:
        try:
            rows, _available, _backend = read_events(f, tags=[tag])
        except Exception as e:
            # Skip files that can't be parsed
            print(f"[Warn] Skip file due to parse error: {f}: {e}")
            continue
        for r in rows:
            v = r.get("value")
            if v is None:
                continue
            if (max_value is None) or (v > max_value):
                max_value = v
                max_step = r.get("step")
                max_file = os.path.basename(f)

    if max_value is None:
        return None

    return {
        "run": os.path.basename(run_dir.rstrip(os.sep)),
        "max_value": float(max_value),
        "max_step": int(max_step) if max_step is not None else None,
        "max_file": max_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect max of a scalar tag from each run directory under logs root.")
    parser.add_argument("logs_root", type=str, help="目录，包含若干实验子目录（每个子目录内有 events.out.tfevents.*）")
    parser.add_argument("--tag", type=str, default="valid_acc_1", help="要提取的标量 tag，默认 valid_acc_1")
    parser.add_argument("--out", type=str, default=None, help="输出 CSV 路径；默认保存在 logs_root 下，文件名为 max_<tag>.csv")
    parser.add_argument("--latest", action="store_true", help="每个子目录只读取最新一个事件文件（默认全读）")
    args = parser.parse_args()

    logs_root = args.logs_root
    tag = args.tag

    if not os.path.isdir(logs_root):
        raise SystemExit(f"logs_root not found or not a directory: {logs_root}")

    subdirs = [os.path.join(logs_root, d) for d in sorted(os.listdir(logs_root))
               if os.path.isdir(os.path.join(logs_root, d))]

    results: List[Dict[str, Any]] = []
    for sd in subdirs:
        r = compute_max_for_run(sd, tag, latest_only=args.latest)
        if r is None:
            print(f"[Info] No data for tag '{tag}' in run: {os.path.basename(sd)}")
            continue
        results.append(r)

    if not results:
        print(f"No data found for tag '{tag}' under: {logs_root}")
        return

    out_path = args.out
    if out_path is None:
        safe_tag = tag.replace('/', '_')
        out_path = os.path.join(logs_root, f"max_{safe_tag}.csv")

    # Try pandas first for nicer output; fall back to csv module
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(results)
        df = df.sort_values(by=["run"])  # stable order
        df.to_csv(out_path, index=False)
        print(f"[OK] Saved {len(df)} rows to: {out_path}")
    except Exception as e:
        print(f"[Warn] pandas not available or failed ({e}); using csv module.")
        import csv
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["run", "max_value", "max_step", "max_file"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"[OK] Saved {len(results)} rows to: {out_path}")


if __name__ == "__main__":
    main()

