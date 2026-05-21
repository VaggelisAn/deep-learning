#!/usr/bin/env python3
"""Parse NeuMF training logs for per-iteration and best-session metrics.

Usage:
    python3 parse_training_logs.py logs/sample_log.txt
    python3 parse_training_logs.py logs/*.txt

Output:
    - per-session best HR/NDCG
    - per-iteration HR/NDCG counts
    - mean and standard deviation of best HR and NDCG
"""

import argparse
import glob
import math
import os
import re
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BestMetric = Tuple[float, float]

SESSION_SPLIT_RE = re.compile(r'(?m)^-+\s*$')
BEST_RE = re.compile(r'End\. Best Iteration \d+:\s+HR = ([0-9.]+), NDCG = ([0-9.]+)\.')
ITER_RE = re.compile(r'Iteration\s+\d+\s+\[[0-9.]+\s*s\]:\s+HR = ([0-9.]+), NDCG = ([0-9.]+),')


def parse_sessions(text: str) -> List[str]:
    parts = SESSION_SPLIT_RE.split(text)
    return [p.strip() for p in parts if 'End. Best Iteration' in p]


def parse_best_metrics(session: str) -> Optional[BestMetric]:
    match = BEST_RE.search(session)
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)))


def parse_iteration_metrics(session: str) -> List[BestMetric]:
    return [(float(hr), float(ndcg)) for hr, ndcg in ITER_RE.findall(session)]


def load_paths(paths: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for path in paths:
        expanded.extend(sorted(glob.glob(path)))
    return expanded


def summarize(results: Sequence[BestMetric]) -> Dict[str, float]:
    if not results:
        return {'mean': 0.0, 'std': 0.0, 'count': 0}
    values = [r[0] for r in results]
    return {
        'mean': statistics.mean(values),
        'std': statistics.pstdev(values) if len(values) > 1 else 0.0,
        'count': len(values),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Parse training logs for HR/NDCG metrics.')
    parser.add_argument('paths', nargs='+', help='One or more log file paths or glob patterns.')
    parser.add_argument('--show-iterations', action='store_true', help='Print per-iteration metrics for each session.')
    args = parser.parse_args(argv)

    files = load_paths(args.paths)
    if not files:
        print('No files matched the provided patterns.')
        return 1

    all_best: List[BestMetric] = []
    session_index = 0

    for file_path in files:
        if not os.path.isfile(file_path):
            print(f'Skipping missing file: {file_path}')
            continue
        text = open(file_path, 'r', encoding='utf-8', errors='ignore').read()
        sessions = parse_sessions(text)
        if not sessions:
            print(f'No valid sessions found in {file_path}')
            continue

        for session_no, session_text in enumerate(sessions, start=1):
            session_index += 1
            best = parse_best_metrics(session_text)
            iterations = parse_iteration_metrics(session_text)
            print(f'File: {file_path} | Session: {session_no} | Iterations: {len(iterations)}')
            if best:
                print(f'  Best HR = {best[0]:.6f}, Best NDCG = {best[1]:.6f}')
                all_best.append(best)
            else:
                print('  Warning: no best metric found for this session.')
            if args.show_iterations:
                for i, (hr, ndcg) in enumerate(iterations, start=0):
                    print(f'    Iter {i}: HR = {hr:.6f}, NDCG = {ndcg:.6f}')

    if all_best:
        hrs = [hr for hr, _ in all_best]
        ndcgs = [ndcg for _, ndcg in all_best]
        print('\nAggregate best-session metrics:')
        print(f'  sessions: {len(all_best)}')
        print(f'  best HR mean = {statistics.mean(hrs):.6f}')
        print(f'  best HR std  = {statistics.pstdev(hrs) if len(hrs) > 1 else 0.0:.6f}')
        print(f'  best NDCG mean = {statistics.mean(ndcgs):.6f}')
        print(f'  best NDCG std  = {statistics.pstdev(ndcgs) if len(ndcgs) > 1 else 0.0:.6f}')
    else:
        print('No best-session metrics parsed from input files.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
