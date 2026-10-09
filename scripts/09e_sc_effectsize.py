# -*- coding: utf-8 -*-
"""
09e 单细胞效应量分析（审稿意见 #7）
用官方标准化矩阵（sc_normalized_check.npz，与稿件 Fig7 统计量已逐位核对），
对肝细胞 Tumor vs Normal 补充：中位数、倍数变化、Cliff's delta、检出率差异，
避免"超大样本下微小差异极高显著"的误读。
输出：results/rev7_sc_effectsize.csv, figs/FigS7_sc_effectsize.pdf
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, RAW, FIG_DPI, COL_RISK, COL_SAFE
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

GENES = ["SFPQ", "HNRNPM", "IGF2BP3", "NOVA1"]


def cliffs_delta(x, y):
    """Cliff's delta = 2*P(x>y) - 1（用 Mann-Whitney U 计算，ties 近似）"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    U = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2 * U / (n1 * n2) - 1


def main():
    off = np.load(f"{OUT}/sc_normalized_check.npz")
    cells = list(off["cells"])
    meta = pd.read_csv(os.path.join(RAW, "GSE149614_HCC.metadata.updated.txt.gz"),
                       sep="\t").set_index("Cell")
    common = [c for c in cells if c in meta.index]
    m = meta.loc[common]
    E = pd.DataFrame({g: off[g] for g in GENES}, index=cells).loc[common].T
    coef = pd.read_csv(f"{OUT}/model_genes.csv").set_index("gene")["coef"]
    z = E.sub(E.mean(axis=1), axis=0).div(E.std(axis=1).replace(0, np.nan), axis=0)
    risk = sum(z.loc[g] * coef.get(g, 0) for g in GENES)

    hepa = m.celltype == "Hepatocyte"
    mh = m[hepa]
    tum, nor = mh.site == "Tumor", mh.site == "Normal"

    rows = []
    for g in GENES + ["risk_score"]:
        v = risk if g == "risk_score" else E.loc[g]
        vt, vn = v[hepa][tum], v[hepa][nor]
        p = stats.mannwhitneyu(vt, vn).pvalue
        fc = float(vt.mean() / vn.mean()) if (g != "risk_score" and vn.mean() > 1e-9) else np.nan
        rows.append((g, int(tum.sum()), int(nor.sum()),
                     float(vn.median()), float(vt.median()),
                     float(vt.median() - vn.median()), fc,
                     cliffs_delta(vt, vn), p,
                     float((vt > 0).mean() * 100), float((vn > 0).mean() * 100)))
    tab = pd.DataFrame(rows, columns=["gene", "n_tumor", "n_normal", "median_normal", "median_tumor",
                                      "median_diff", "fold_change_mean", "cliffs_delta", "MWU_p",
                                      "pct_expr_tumor", "pct_expr_normal"])
    tab.to_csv(f"{OUT}/rev7_sc_effectsize.csv", index=False)
    print("肝细胞 Tumor vs Normal 效应量:")
    print(tab.to_string(index=False))

    # 各主要细胞类型的中位数（背景）
    bg = []
    for ct in ["Hepatocyte", "Myeloid", "T/NK", "Endothelial", "Fibroblast", "B"]:
        sel = m.celltype == ct
        if sel.sum() == 0:
            continue
        for g in GENES:
            bg.append((ct, g, sel.sum(), float(E.loc[g, sel].median()),
                       float((E.loc[g, sel] > 0).mean() * 100)))
    pd.DataFrame(bg, columns=["celltype", "gene", "n_cells", "median_expr", "pct_expr"]
                 ).to_csv(f"{OUT}/rev7_sc_celltype_median.csv", index=False)

    # 图：肝细胞各基因/评分的小提琴 + 中位数标注
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.8), sharey=False)
    for ax, g in zip(axes, GENES + ["risk_score"]):
        v = risk if g == "risk_score" else E.loc[g]
        data = [v[hepa][nor], v[hepa][tum]]
        parts = ax.violinplot(data, widths=0.7, showextrema=False)
        for pc, c in zip(parts["bodies"], [COL_SAFE, COL_RISK]):
            pc.set_facecolor(c); pc.set_alpha(0.75)
        med = [np.median(data[0]), np.median(data[1])]
        ax.scatter([1, 2], med, c="black", s=25, zorder=3)
        ax.plot([1, 2], med, c="black", lw=1, ls="--")
        cd = tab.set_index("gene").loc[g, "cliffs_delta"]
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Normal", "Tumor"], fontsize=9)
        ax.set_title(f"{g}\nCliff's δ={cd:.3f}", fontsize=9)
        ax.set_ylabel("Normalized expression" if g == GENES[0] else ("Risk score" if g == "risk_score" else ""))
    fig.suptitle("Hepatocytes, tumor vs normal (GSE149614): effect sizes with medians", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/FigS7_sc_effectsize.pdf", dpi=FIG_DPI); plt.close(fig)
    print("\n09e 完成")


if __name__ == "__main__":
    main()
