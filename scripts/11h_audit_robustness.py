# -*- coding: utf-8 -*-
"""
11h 三审 Major1/2 整改：探针拆分偏倚审计 + 同源对照 + Immune141 重叠度
输出：
  results/rev14_probe_split.csv     每个筛选池基因的探针唯一性标注
  results/rev14_audit_robust.txt    计数核对 + 唯一探针子集上调比例 + 同源对照
  results/rev14_immune_overlap.txt  Bindea/Charoentong 细胞集与 Immune141 基因重叠
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, RAW

aud = pd.read_csv(f"{OUT}/rev9_gene_audit.csv")

# ---- 1. 探针拆分标注 ----
xl = pd.ExcelFile(os.path.join(RAW, "Li2017_mmc3.xlsx"))
probe_genes = {}   # gene -> 出现的 probe 行集合
shared = set()     # 出现在任一多基因行中的基因
for sheet in ("ds-egfp v.s ss-egfp", "IRAlu v.s Alu"):
    df = xl.parse(sheet)
    for _, row in df.iterrows():
        v = row["Target gene names"]
        if pd.isna(v):
            continue
        genes = [g.strip() for g in str(v).split(";")
                 if g.strip() and not g.strip().startswith("hCG") and g.strip() != "nan"]
        if not genes:
            continue
        for g in genes:
            probe_genes.setdefault(g, set()).add((sheet, row.name))
        if len(genes) > 1:
            shared.update(genes)
pool = set(probe_genes)
aud["probe_status"] = aud["gene"].map(
    lambda g: ("screen_pool" if g in pool else "literature") )
aud["probe_unique"] = aud["gene"].map(
    lambda g: ("shared_multi_gene_probe" if g in shared else
               ("unique_single_gene_probe" if g in pool else "n/a_literature")))
aud.to_csv(f"{OUT}/rev14_probe_split.csv", index=False)

n_split = len(shared & pool)
print(f"screen pool total: {len(pool)}; from multi-gene probe rows: {n_split} ({n_split/len(pool):.1%})")

# ---- 2. 计数核对 + 同源对照 + 唯一探针稳健性 ----
det = aud[aud.in_expr_matrix == True]
scr_all = aud[aud.probe_status == "screen_pool"]                 # 全部筛选池基因
scr_cur = scr_all[scr_all.in_final_60]                           # 9 个已纳入
scr_non = scr_all[~scr_all.in_final_60]                          # 未纳入（含未映射）
scr_non_det = det[(det.probe_status == "screen_pool") & (~det.in_final_60)]  # 141
fin = det[det.in_final_60]

def uprate(d):
    return f"{(d.DE_sig == 'Up').mean():.1%} (n={len(d)})"

lines = [
    f"screen pool unique symbols: {len(scr_all)}",
    f"curated into final 60 (from screen): {len(scr_cur)}",
    f"not curated: {len(scr_non)} total; measurable in TCGA: {len(scr_non_det)}; not mapped: {len(scr_non) - len(scr_non_det)}",
    f"upregulated rate (FDR<0.05, log2FC>0):",
    f"  non-curated screen genes (measurable): {uprate(scr_non_det)}",
    f"  curated screen hits (same-source control): {uprate(det[det.gene.isin(scr_cur.gene)])}",
    f"  final 60 (all): {uprate(fin)}",
    "",
    "robustness: unique-single-gene-probe subsets only",
    f"  non-curated screen, unique probes: {uprate(scr_non_det[scr_non_det.probe_unique == 'unique_single_gene_probe'])}",
    f"  curated screen hits, unique probes: {uprate(det[(det.gene.isin(scr_cur.gene)) & (det.probe_unique == 'unique_single_gene_probe')])}",
]
open(f"{OUT}/rev14_audit_robust.txt", "w").write("\n".join(lines))
print("\n".join(lines))

# ---- 3. Immune141 与免疫细胞基因集重叠 ----
est = pd.read_excel(os.path.join(RAW, "ESTIMATE_moesm488.xlsx"), sheet_name="gene", header=1)
imm = set(est.loc[est["Set"] == "Immune141_UP", "Gene"].dropna().astype(str))
gmt = {}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "genesets", "immune_cells.gmt"), encoding="utf-8") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        gmt[parts[0]] = set(parts[2:])
all_imm_cells = set().union(*gmt.values())
ol = imm & all_imm_cells
out2 = [
    f"Immune141 genes: {len(imm)}",
    f"union of 19 immune cell marker sets: {len(all_imm_cells)}",
    f"overlap (Immune141 ∩ cell-set union): {len(ol)} ({len(ol)/len(imm):.1%} of Immune141; {len(ol)/len(all_imm_cells):.1%} of cell-set union)",
]
open(f"{OUT}/rev14_immune_overlap.txt", "w").write("\n".join(out2))
print("\n".join(out2))
print("\n11h 完成")
