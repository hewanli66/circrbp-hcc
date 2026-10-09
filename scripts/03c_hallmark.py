# -*- coding: utf-8 -*-
"""
03c 风险评分与 Hallmark 通路活性关联（补充 Fig3D）
  - 从 Enrichr 获取 MSigDB Hallmark 2020 基因集
  - ssGSEA 计算每条通路活性，与风险评分做 Spearman 相关
输出：figs/Fig3D_hallmark_corr.pdf, results/hallmark_corr.csv
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

from ssgea import ssgsea_from_gmt  # 本地工具函数


def main():
    import gseapy as gp
    expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
    risk = pd.read_csv(f"{OUT}/risk_scores.csv").set_index("sample")["risk_score"]
    risk = risk[~risk.index.duplicated()]
    expr = expr[risk.index]

    lib = gp.get_library("MSigDB_Hallmark_2020", organism="human")
    print(f"Hallmark 通路数: {len(lib)}")
    scores = ssgsea_from_gmt(expr, lib)
    scores.to_csv(f"{OUT}/hallmark_ssgsea.csv")

    rows = []
    for p in scores.index:
        r, pv = stats.spearmanr(scores.loc[p], risk)
        rows.append((p.replace("HALLMARK-", ""), r, pv))
    df = pd.DataFrame(rows, columns=["pathway", "rho", "p"]).sort_values("rho")
    df["fdr"] = stats.false_discovery_control(df["p"].clip(lower=1e-300))
    df.to_csv(f"{OUT}/hallmark_corr.csv", index=False)

    top = pd.concat([df.head(6), df.tail(6)])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = ["#4DBBD5" if r < 0 else "#E64B35" for r in top.rho]
    ax.barh(top.pathway, top.rho, color=colors)
    ax.set_xlabel("Spearman correlation with circRBP risk score")
    ax.axvline(0, c="k", lw=.8)
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig3D_hallmark_corr.pdf", dpi=300); plt.close(fig)
    print("正相关最强:", df.tail(4)["pathway"].tolist())
    print("负相关最强:", df.head(4)["pathway"].tolist())
    print("03c 完成")


if __name__ == "__main__":
    main()
