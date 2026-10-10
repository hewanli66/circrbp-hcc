# -*- coding: utf-8 -*-
"""
11b 基因集审计（二审 Major 2 整改）
重建 103->60 的完整筛选链，并对全部筛选池基因计算 TCGA 表达/DE/Cox 统计：
  池1: Li et al. 2017 全基因组 siRNA 筛选命中（补充表 mmc3 两个 reporter 比较的并集）
  池2: 文献整理的 circRNA 生物发生/剪接调控 RBP（我方最终60中不在筛选命中内的部分）
输出：
  results/rev9_gene_audit.csv        全部候选基因的统计+来源+是否入最终60
  results/rev9_audit_summary.txt     汇总（供 Methods/回复信引用）
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, RAW
from scipy import stats
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests

# ---- 1. 筛选池 ----
xl = pd.ExcelFile(os.path.join(RAW, "Li2017_mmc3.xlsx"))
def split_genes(series):
    out = set()
    for v in series.dropna():
        for g in str(v).split(";"):
            g = g.strip()
            if g and not g.startswith("hCG") and g != "nan":
                out.add(g)
    return out
g_ds = split_genes(xl.parse("ds-egfp v.s ss-egfp")["Target gene names"])
g_ir = split_genes(xl.parse("IRAlu v.s Alu")["Target gene names"])
screen_pool = sorted((g_ds | g_ir))

final60 = pd.read_csv(f"{OUT}/circrbp_deg.csv")["gene"].tolist()
final_set = set(final60)
lit_add = sorted(final_set - (g_ds | g_ir))          # 文献增补
in_screen = sorted(final_set & (g_ds | g_ir))        # 来自筛选
print(f"screen pool (union, split): {len(screen_pool)}; final60 in screen: {len(in_screen)}; literature: {len(lit_add)}")

# ---- 2. TCGA 统计 ----
expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
clin = pd.read_csv(f"{OUT}/tcga_clin.csv")
bar = pd.Series(expr.columns).str[13:15].values
tum_cols = expr.columns[bar == "01"]
nor_cols = expr.columns[bar == "11"]

candidates = sorted(set(screen_pool) | final_set)
rows = []
for g in candidates:
    if g not in expr.index:
        rows.append(dict(gene=g, in_expr_matrix=False))
        continue
    t = expr.loc[g, tum_cols].dropna().astype(float)
    n = expr.loc[g, nor_cols].dropna().astype(float)
    rec = dict(gene=g, in_expr_matrix=True, mean_tumor=round(t.mean(), 4), mean_normal=round(n.mean(), 4))
    if len(t) >= 3 and len(n) >= 3 and t.std() > 0:
        rec["log2FC"] = round(t.mean() - n.mean(), 4)
        rec["p"] = stats.ttest_ind(t, n, equal_var=False).pvalue
    rows.append(rec)

aud = pd.DataFrame(rows)
aud["FDR"] = np.nan
mask = aud["p"].notna()
aud.loc[mask, "FDR"] = multipletests(aud.loc[mask, "p"], method="fdr_bh")[1]
aud["DE_sig"] = np.where(aud.FDR < 0.05, np.where(aud.log2FC > 0, "Up", "Down"),
                         np.where(aud.FDR < 0.05, "Down", "ns"))

# Cox（对可检测基因）
surv = clin.set_index("sample")[["OS", "OS.time"]]
sub = expr.loc[expr.index.intersection([g for g in candidates if g in expr.index])]
dat = sub.T.loc[surv.index]
cox_rows = []
for g in dat.columns:
    d = pd.concat([dat[g].astype(float), surv], axis=1).dropna()
    d.columns = ["x", "OS", "OS.time"]
    if d["x"].std() == 0 or len(d) < 50:
        continue
    try:
        cph = CoxPHFitter().fit(d, "OS.time", "OS", show_progress=False)
        ci = cph.confidence_intervals_.values[0]
        cox_rows.append(dict(gene=g, HR=float(cph.hazard_ratios_["x"]),
                             HR_p=float(cph.summary.p.iloc[0])))
    except Exception:
        pass
cox = pd.DataFrame(cox_rows)
aud = aud.merge(cox, on="gene", how="left")

# ---- 3. 来源与入组标注 ----
def source(g):
    s = []
    if g in g_ds: s.append("Li2017_screen_ds-egfp")
    if g in g_ir: s.append("Li2017_screen_IRAlu")
    if g in lit_add: s.append("literature-curated")
    return ";".join(s) if s else "screen_hit_not_curated"
aud["source"] = aud["gene"].map(source)
aud["in_final_60"] = aud["gene"].isin(final_set)
aud = aud[["gene", "source", "in_final_60", "in_expr_matrix", "mean_tumor", "mean_normal",
           "log2FC", "p", "FDR", "DE_sig", "HR", "HR_p"]]
aud = aud.sort_values(["in_final_60", "gene"], ascending=[False, True])
aud.to_csv(f"{OUT}/rev9_gene_audit.csv", index=False)

# ---- 4. 汇总：选择偏倚审计 ----
det = aud[aud.in_expr_matrix == True]
scr = det[~det.in_final_60 & det.source.str.contains("Li2017")]   # 筛选命中但未入60
fin = det[det.in_final_60]
def uprate(d):
    return (d.DE_sig == "Up").mean(), len(d)
u_scr, n_scr = uprate(scr)
u_fin, n_fin = uprate(fin)
u_all60_raw = (fin.DE_sig == "Up").mean()
lines = [
    f"screen pool union (split genes): {len(screen_pool)}",
    f"final60: {len(final_set)} (from screen: {len(in_screen)}; literature-curated: {len(lit_add)})",
    f"screen hits mapped in TCGA expr matrix: {scr.shape[0]} (not curated into 60)",
    f"upregulated rate (FDR<0.05, log2FC>0):",
    f"  non-curated screen genes: {u_scr:.1%} (n={n_scr})",
    f"  final 60 genes: {u_all60_raw:.1%} (n={n_fin})",
]
open(f"{OUT}/rev9_audit_summary.txt", "w").write("\n".join(lines))
print("\n".join(lines))
print("\n11b 完成")
