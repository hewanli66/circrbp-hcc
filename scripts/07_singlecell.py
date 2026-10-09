# -*- coding: utf-8 -*-
"""
07 单细胞验证（GSE149614）
数据：计数矩阵 GSE149614_HCC.scRNAseq.S71915.count.txt.gz（基因 x 71915 细胞）
      元数据 GSE149614_HCC.metadata.updated.txt.gz（Cell/sample/site/patient/stage/virus/celltype）
内存策略：分块读取，第一遍累计每个细胞的总计数，第二遍提取 4 个模型基因
输出：figs/Fig7A_sc_dotplot.pdf, figs/Fig7B_hepatocyte_site.pdf
      results/sc_gene_stats.csv, results/sc_hepatocyte_stats.csv
"""
import sys, os, gzip
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, RAW, FIG_DPI, COL_RISK, COL_SAFE
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

GENES = ["SFPQ", "HNRNPM", "IGF2BP3", "NOVA1"]
COUNT = os.path.join(RAW, "GSE149614_HCC.scRNAseq.S71915.count.txt.gz")
META  = os.path.join(RAW, "GSE149614_HCC.metadata.updated.txt.gz")
CHUNK = 300


def main():
    meta = pd.read_csv(META, sep="\t")
    print(f"元数据: {len(meta)} 细胞, 类型: {meta.celltype.nunique()}")
    meta = meta.set_index("Cell")

    # ---------- 单遍流式读取：列计数累计 + 提取 4 基因 ----------
    gene_rows = {}
    t0 = __import__("time").time()
    with gzip.open(COUNT, "rt") as f:
        cells = f.readline().rstrip("\n").split("\t")
        n = len(cells)
        sums = np.zeros(n, dtype=np.float64)
        n_genes = 0
        for line in f:
            parts = line.split("\t")
            if parts[0] in GENES and parts[0] not in gene_rows:
                gene_rows[parts[0]] = np.array(parts[1:], dtype=np.float32)
            sums += np.fromiter(parts[1:], dtype=np.float32, count=n)
            n_genes += 1
    print(f"计数矩阵: {n_genes} 基因 x {n} 细胞, 读取耗时 {__import__('time').time()-t0:.0f}s")
    print(f"提取到: {sorted(gene_rows)}")
    expr = pd.DataFrame(gene_rows, index=cells).T   # 4 x n_cells（行=基因, 列=细胞）
    expr = expr[~expr.index.duplicated(keep="first")]
    tot = pd.Series(sums, index=cells)
    norm = np.log2(1 + expr.div(tot / 1e4))         # scanpy 风格 log 归一化
    print(f"细胞匹配: 表达矩阵 {expr.shape[1]} 列, 元数据匹配 {meta.index.isin(expr.columns).sum()}")

    # 对齐
    common = [c for c in expr.columns if c in meta.index]
    norm = norm[common]
    m = meta.loc[common]

    stats_rows = []
    for ct in m.celltype.unique():
        cells = m.celltype == ct
        for g in GENES:
            v = norm.loc[g, cells]
            stats_rows.append((ct, g, v.mean(), (v > 0).mean() * 100, cells.sum()))
    st = pd.DataFrame(stats_rows, columns=["celltype", "gene", "mean_expr", "pct_expr", "n_cells"])
    st.to_csv(f"{OUT}/sc_gene_stats.csv", index=False)

    # ---------- Fig7A: 点图 ----------
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    cts = ["Hepatocyte", "Myeloid", "T/NK", "B", "Endothelial", "Fibroblast"]
    cts = [c for c in cts if c in st.celltype.unique()]
    for y, g in enumerate(GENES):
        for x, ct in enumerate(cts):
            r = st[(st.celltype == ct) & (st.gene == g)]
            if not len(r):
                continue
            mean_e, pct = r.mean_expr.iloc[0], r.pct_expr.iloc[0]
            ax.scatter(x, y, s=pct * 6, c=[mean_e], cmap="RdYlBu_r", vmin=0,
                       vmax=max(3.0, st.mean_expr.max()), edgecolors="grey", lw=0.5, zorder=3)
    ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(GENES))); ax.set_yticklabels(GENES, fontsize=10)
    ax.set_xlabel("")
    sc = ax.scatter([], [], c=[], cmap="RdYlBu_r", vmin=0, vmax=max(3.0, st.mean_expr.max()))
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Mean log2 norm. expr.", fontsize=8)
    for pct, lab in [(20, "20%"), (60, "60%")]:
        ax.scatter([], [], s=pct * 6, facecolors="none", edgecolors="grey", label=lab)
    ax.legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), frameon=False, fontsize=8,
              title="Dot = % expressing", title_fontsize=8)
    ax.set_title("Single-cell expression of signature genes (GSE149614)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig7A_sc_dotplot.pdf", dpi=FIG_DPI); plt.close(fig)

    # ---------- Fig7B: 肝细胞按部位比较 + 单细胞风险分 ----------
    hepa = m.celltype == "Hepatocyte"
    mh = m[hepa]
    eh = norm.loc[:, hepa]                    # 4 x n_hepatocytes
    # 单细胞风险分：4 基因 z 值加权和（系数沿用 bulk 模型）
    coef = pd.read_csv(f"{OUT}/model_genes.csv").set_index("gene")["coef"]
    z = eh.sub(eh.mean(axis=1), axis=0).div(eh.std(axis=1).replace(0, np.nan), axis=0)
    rs = sum(z.loc[g] * coef.get(g, 0) for g in GENES if g in z.index)
    order = [s for s in ["Normal", "Tumor", "PVTT", "Lymph"] if s in mh.site.unique()]
    rows = []
    for g in GENES:
        fig, axes = plt.subplots(1, 5, figsize=(13, 3.2),
                                 gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.2]})
        for ax, gg in zip(axes[:4], GENES):
            data = [eh.loc[gg, mh.site == s].dropna() for s in order]
            bp = ax.boxplot(data, labels=order, patch_artist=True, widths=0.6, showfliers=False)
            for p, s in zip(bp["boxes"], order):
                p.set_facecolor("#4DBBD5" if s == "Normal" else ("#E64B35" if s == "Tumor" else "#BCBD22"))
            pv = stats.mannwhitneyu(eh.loc[gg, mh.site == "Tumor"].dropna(),
                                    eh.loc[gg, mh.site == "Normal"].dropna()).pvalue if "Tumor" in order and "Normal" in order else np.nan
            ax.set_title(f"{gg}\nT vs N p={pv:.1e}", fontsize=9)
            ax.tick_params(axis="x", rotation=30, labelsize=7)
            ax.set_ylabel("log2 norm. expr." if gg == GENES[0] else "", fontsize=8)
            rows.append((gg, pv))
        ax = axes[4]
        data = [rs[mh.site == s].dropna() for s in order]
        bp = ax.boxplot(data, labels=order, patch_artist=True, widths=0.6, showfliers=False)
        for p, s in zip(bp["boxes"], order):
            p.set_facecolor("#4DBBD5" if s == "Normal" else ("#E64B35" if s == "Tumor" else "#BCBD22"))
        pv = stats.mannwhitneyu(rs[mh.site == "Tumor"].dropna(),
                                rs[mh.site == "Normal"].dropna()).pvalue
        ax.set_title(f"4-gene risk score\nT vs N p={pv:.1e}", fontsize=9)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        fig.suptitle("Hepatocytes across tissue sites (GSE149614)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(f"{FIG}/Fig7B_hepatocyte_site.pdf", dpi=FIG_DPI); plt.close(fig)
        break  # 只出一版（4基因+风险分并列）

    hp = pd.DataFrame(rows, columns=["gene", "Tumor_vs_Normal_p"])
    hp.to_csv(f"{OUT}/sc_hepatocyte_stats.csv", index=False)
    print("肝细胞中 Tumor vs Normal 检验:")
    print(hp.to_string(index=False))
    print("07 完成")


if __name__ == "__main__":
    main()
