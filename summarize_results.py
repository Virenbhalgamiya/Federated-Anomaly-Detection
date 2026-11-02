"""Merge results/results*.csv (one per parallel process) and write RESULTS.md and the charts."""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("SEEDS", "0")  # run_experiments reads this at import; unused here
from run_experiments import summarize

files = sorted(glob.glob("results/results*.csv"))
df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(subset=["seed", "partition", "method"])
print(f"merged {len(files)} files, {len(df)} runs, seeds {sorted(df.seed.unique())}")
summarize(df, open("results/dataset_stats.txt").read().strip())
