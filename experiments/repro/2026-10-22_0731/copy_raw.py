"""Copy exactly the raw files listed in data/MANIFEST.csv from the main repo into a clone.

Reads from the main repo only; writes only inside the clone. Threads overlap disk reads/writes.
"""

import csv
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

main, clone = Path(sys.argv[1]), Path(sys.argv[2])
with open(main / "data" / "MANIFEST.csv", newline="", encoding="utf-8") as f:
    files = [row["file"] for row in csv.DictReader(f)]


def copy(rel: str) -> int:
    src, dst = main / "data" / "raw" / rel, clone / "data" / "raw" / rel
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


t0 = time.time()
with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
    total = sum(pool.map(copy, files))
print(f"copied {len(files)} files, {total / 1e9:.2f} GB in {time.time() - t0:.0f} s "
      f"with {os.cpu_count()} threads", flush=True)
