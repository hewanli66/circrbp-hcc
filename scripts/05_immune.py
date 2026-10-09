# -*- coding: utf-8 -*-
"""
05 免疫微环境分析
内容：
  - ssGSEA 计算 28 种免疫细胞浸润（ImmPort/Charoentong 基因集）
  - 高低风险组免疫浸润差异（箱线图+热图）
  - 风险评分与免疫检查点基因表达相关性
  - （可选）TIDE 免疫治疗应答预测：上传结果后解析
输出：figs/Fig5A_immune_box.pdf, Fig5B_checkpoint_corr.pdf, results/immune_ssgsea.csv
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, GENESET, COL_GROUP
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

CHECKPOINTS = ["PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "PDCD1LG2",
               "CD80", "CD86", "IDO1", "ICOS", "ICOSLG"]


def ssgses(expr, gmt):
    """简化 ssGSEA：对每个基因集按基因表达秩次积分"""
    from scipy.stats import rankdata
    E = expr.values
    ranks = np.apply_along_axis(rankdata, 0, E)
    out = {}
    n = E.shape[0]
    for name, genes in gmt.items():
        hit = np.array([g in expr.index for g in genes])
        present = [g for g in genes if g in expr.index]
        if len(present) < 3: continue
        idx = [list(expr.index).index(g) for g in present]
        P = ranks[idx, :]
        # 积分：命中基因秩次的相对积分
        score = (np.abs(P - (n + 1) / 2)).sum(axis=0) / len(idx)
        out[name] = score
    return pd.DataFrame(out, index=expr.columns).T


def load_gmt(path):
    gmt = {}
    for line in open(path, encoding="utf-8"):
        parts = line.strip().split("\t")
        if len(parts) > 2:
            gmt[parts[0]] = parts[2:]
    return gmt


def main():
    expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
    risk = pd.read_csv(f"{OUT}/risk_scores.csv").set_index("sample")["risk_score"]
    risk = risk[~risk.index.duplicated()]
    expr = expr[risk.index]

    gmt = load_gmt(f"{GENESET}/immune_cells.gmt")
    s = ssgses(expr, gmt)
    s.to_csv(f"{OUT}/immune_ssgsea.csv")
    grp = (risk > risk.median()).map({True: "High", False: "Low"})

    # 差异箱线
    pv = []
    for c in s.index:
        a = s.loc[c, grp == "High"]; b = s.loc[c, grp == "Low"]
        pv.append(stats.mannwhitneyu(a, b).pvalue)
    pv = np.array(pv); order = np.argsort(pv)
    top = s.index[order][:15]
    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    for ax, c in zip(axes.flat, top):
        d = pd.DataFrame({"group": grp, "v": s.loc[c]})
        for i, (g, col) in enumerate([("High", COL_GROUP[1]), ("Low", COL_GROUP[0])]):
            ax.boxplot(d.loc[d.group == g, "v"], positions=[i], widths=0.5,
                       boxprops=dict(color=col), medianprops=dict(color="k"))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["High", "Low"])
        ax.set_title(f"{c}\np={pv[list(s.index).index(c)]:.1e}", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig5A_immune_box.pdf", dpi=300); plt.close(fig)

    # 检查点相关性
    ck = expr.loc[expr.index.intersection(CHECKPOINTS)]
    cor = ck.apply(lambda r: stats.spearmanr(r, risk).statistic, axis=1)
    pck = ck.apply(lambda r: stats.spearmanr(r, risk).pvalue, axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = np.where(cor > 0, COL_GROUP[1], COL_GROUP[0])
    ax.barh(ck.index, cor, color=colors)
    ax.set_xlabel("Spearman correlation with risk score")
    for i, (c, p) in enumerate(zip(cor, pck)):
        ax.text(c + (0.01 if c > 0 else -0.01), i, f"p={p:.1e}", va="center",
                ha="left" if c > 0 else "right", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig5B_checkpoint_corr.pdf", dpi=300); plt.close(fig)
    print("05 完成；免疫浸润差异显著的细胞类型数：", (pv < 0.05).sum())


if __name__ == "__main__":
    main()
