# -*- coding: utf-8 -*-
"""
02 差异表达 + 单因素Cox筛选（针对 circRBP 基因集）
输出：
  figs/Fig1A_volcano.pdf, Fig1B_heatmap.pdf     circRBP 在肿瘤vs正常的表达差异
  figs/Fig1C_forest.pdf                          单因素 Cox 森林图
  results/circrbp_deg.csv, results/univ_cox.csv
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, CIRCRBP_LIST, FDR, LOGFC, COX_P, COL_GROUP
from scipy import stats
from lifelines import CoxPHFitter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
plt.rcParams["font.family"] = "DejaVu Sans"


def deg(expr, clin):
    """肿瘤 vs 配对/非配对正常。正常样本 barcode 第14-15位为 11"""
    bar = pd.Series(expr.columns).str[13:15].values
    tum = expr.loc[:, bar == "01"]; nor = expr.loc[:, bar == "11"]
    rows = []
    for g in expr.index:
        t, n = tum.loc[g].dropna(), nor.loc[g].dropna()
        if len(t) < 3 or len(n) < 3: continue
        fc = t.mean() - n.mean()
        pv = stats.ttest_ind(t, n, equal_var=False).pvalue
        rows.append((g, fc, pv))
    df = pd.DataFrame(rows, columns=["gene", "log2FC", "p"])
    df["FDR"] = multipletests(df["p"], method="fdr_bh")[1]
    # 着色按 FDR（FC 阈值仅用于展示参考线）；小基因集普遍上调，FDR 更合理
    df["sig"] = np.where(df.FDR < FDR,
                         np.where(df.log2FC > 0, "Up", "Down"), "NS")
    return df, tum, nor


def volcano(df):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for lab, col in [("Up", COL_GROUP[1]), ("Down", COL_GROUP[0]), ("NS", "grey")]:
        d = df[df.sig == lab]; ax.scatter(d.log2FC, -np.log10(d.p + 1e-300), s=12, c=col, label=lab)
    ax.axvline(LOGFC, ls="--", c="k", lw=.6); ax.axvline(-LOGFC, ls="--", c="k", lw=.6)
    ax.set_xlabel("log2 fold change"); ax.set_ylabel("-log10(p value)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{FIG}/Fig1A_volcano.pdf", dpi=300); plt.close(fig)


def univ_cox(expr, clin):
    """对差异 circRBP 做单因素 Cox"""
    dat = expr.T.loc[clin["sample"]]
    surv = clin.set_index("sample")[["OS", "OS.time"]]
    rows = []
    for g in dat.columns:
        d = pd.concat([dat[g], surv], axis=1).dropna()
        d.columns = ["x", "OS", "OS.time"]
        if d["x"].std() == 0: continue
        try:
            cph = CoxPHFitter().fit(d, "OS.time", "OS", show_progress=False)
            ci = cph.confidence_intervals_.values[0]   # log 空间置信区间
            rows.append((g, float(cph.hazard_ratios_["x"]),
                         float(np.exp(min(ci))), float(np.exp(max(ci))),
                         float(cph.summary.p.iloc[0])))
        except Exception: pass
    res = pd.DataFrame(rows, columns=["gene", "HR", "HR_95L", "HR_95H", "p"])
    res = res.sort_values("p")
    return res[res.p < COX_P]


def forest(res, n=30):
    d = res.head(n)
    fig, ax = plt.subplots(figsize=(6, max(3, n * 0.25)))
    ax.errorbar(d.HR, range(len(d))[::-1], xerr=[d.HR - d.HR_95L, d.HR_95H - d.HR],
                fmt="s", c="k", ecolor="grey", capsize=2)
    ax.axvline(1, ls="--", c="r", lw=.8)
    ax.set_yticks(range(len(d))[::-1]); ax.set_yticklabels(d.gene)
    ax.set_xscale("log"); ax.set_xlabel("Hazard ratio (log scale)")
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig1C_forest.pdf", dpi=300); plt.close(fig)


def main():
    expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
    clin = pd.read_csv(f"{OUT}/tcga_clin.csv")
    genes = [l.strip() for l in open(CIRCRBP_LIST, encoding="utf-8") if l.strip() and not l.startswith("#")]
    sub = expr.loc[expr.index.intersection(genes)]
    print(f"circRBP 基因集中有 {sub.shape[0]}/{len(genes)} 个在表达矩阵中")

    df, tum, nor = deg(sub, clin)
    df.to_csv(f"{OUT}/circrbp_deg.csv", index=False)
    volcano(df)

    # 表达热图（Top 差异基因）
    top = df[df.sig != "NS"].gene.tolist()[:30] or df.head(20).gene.tolist()
    hm = sub.loc[sub.index.isin(top)]
    lab = np.array(["Tumor" if s[13:15] == "01" else "Normal" for s in hm.columns])
    order = np.argsort(lab)
    from seaborn import clustermap
    cg = clustermap(hm.iloc[:, order], cmap="RdBu_r", center=0, col_cluster=False,
                    col_colors=pd.Series(lab[order]).map({"Tumor": COL_GROUP[1], "Normal": COL_GROUP[0]}),
                    figsize=(7, 6), z_score=0)
    cg.savefig(f"{FIG}/Fig1B_heatmap.pdf", dpi=300)

    sig_genes = df[df.sig != "NS"].gene.tolist()
    res = univ_cox(sub.loc[sig_genes], clin)
    res.to_csv(f"{OUT}/univ_cox.csv", index=False)
    forest(res)
    print(f"差异 circRBP: {len(sig_genes)}；其中单因素 Cox 显著: {len(res)} -> {list(res.gene)}")


if __name__ == "__main__":
    main()
