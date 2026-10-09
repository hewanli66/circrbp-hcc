# -*- coding: utf-8 -*-
"""Fast count pass: polars streaming sums + lightweight gene-row scan."""
import gzip, time
import numpy as np, polars as pl

BASE = r"C:\Users\hewan\Documents\kimi\tasks\2026-10-08\11-38-08-f56eebff"
CNT = BASE + r"\rawdata\GSE149614_HCC.scRNAseq.S71915.count.txt.gz"
OUT = BASE + r"\results\sc_count4.npz"
GENES_B = {b"SFPQ", b"HNRNPM", b"IGF2BP3", b"NOVA1"}

t0 = time.time()
with gzip.open(CNT, "rt") as f:
    cells = f.readline().rstrip("\n").split("\t")
n = len(cells)
print("cells:", n, flush=True)

# Q1: per-cell library sums (polars streaming aggregation)
val_cols = [f"column_{i}" for i in range(1, n + 1)]
lf = pl.scan_csv(CNT, separator="\t", has_header=False, skip_rows=1,
                 schema_overrides={c: pl.Int32 for c in val_cols} | {"column_0": pl.Utf8},
                 truncate_ragged_lines=True)
sums = lf.select([pl.sum(c).alias(c) for c in val_cols]).collect().to_numpy().ravel().astype(np.float64)
del lf
print("sums done", f"{time.time()-t0:.0f}s", flush=True)

# Q2: 4 gene rows (lightweight binary scan, no per-line numeric parse)
rows = {}
with gzip.open(CNT, "rb") as f:
    f.readline()
    for line in f:
        tab = line.index(b"\t")
        if line[:tab] in GENES_B:
            g = line[:tab].decode()
            if g not in rows:
                v = np.fromstring(line[tab + 1:].decode(), dtype=np.float32, sep="\t")
                if v.size == n:
                    rows[g] = v
                if len(rows) == 4:
                    break
print("gene rows:", sorted(rows), f"{time.time()-t0:.0f}s", flush=True)

np.savez(OUT, cells=np.array(cells), sums=sums, **rows)
print("saved", OUT, f"{time.time()-t0:.0f}s", flush=True)
