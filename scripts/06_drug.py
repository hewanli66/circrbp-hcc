# -*- coding: utf-8 -*-
"""
06 药物敏感性分析（GDSC1，方案B：风险分-药物敏感性关联分析）
思路：在 GDSC1 细胞系表达矩阵中按 4 基因模型计算风险分，
      对每个药物计算风险分与 LN_IC50 的 Spearman 相关，
      rho>0 提示高风险细胞系耐药，rho<0 提示高风险细胞系敏感。
输出：results/drug_spearman.csv, results/gdsc_risk_score.csv
      figs/Fig6A_drug_landscape.pdf, Fig6B_drug_scatter.pdf, Fig6C_hcc_drug.pdf
"""
import sys, os, glob
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, RAW, GDSC_IC50, COL_RISK, COL_SAFE, FIG_DPI
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

MIN_CELLS = 50       # 每种药物最少细胞系数
MIN_HCC = 8          # HCC 细胞系面板最少细胞系数
HCC_DRUGS = ["Sorafenib", "Lenvatinib", "Regorafenib", "5-Fluorouracil",
             "Gemcitabine", "Doxorubicin", "Erlotinib", "Rapamycin"]


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    fdr = np.empty(n)
    fdr[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.clip(fdr, 0, 1)


def load_gdsc_expression():
    """在 rawdata 中自动识别 GDSC/CMP 表达文件。
    支持两种格式：
      1. 经典 GDSC RMA 文件（Cell_line_RMA_proc_basalExp.txt，基因 x 细胞系）
      2. Cell Model Passports 合并 RNA-seq 文件（含 model_name/data_source/gene_symbol 元数据行）
    """
    cands = []
    for pat in ("*RMA*", "*rma*", "*basalExp*", "*expression*", "*Expression*", "*fpkm*", "*FPKM*"):
        cands += glob.glob(os.path.join(RAW, pat))
    cands = [c for c in cands if not c.endswith(".zip")]
    cands = [c for c in cands if not os.path.basename(c).startswith(("TCGA", "GSE", "TEST"))]
    if not cands:
        return None
    path = sorted(cands)[0]
    print(f"表达文件: {os.path.basename(path)}")
    sep = "\t" if path.endswith((".txt", ".tsv")) else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    # --- CMP 合并 RNA-seq 格式 ---
    if "model_name" in df.index:
        names = df.loc["model_name"].dropna()
        df = df.drop(index=["model_name", "data_source", "gene_symbol"])
        df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
        df.columns = [names.get(c, c) for c in df.columns]   # SIDM -> 细胞系名
        df = df[~df.index.duplicated(keep="first")]
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        return df
    # --- 经典 RMA 格式 ---
    df = df.dropna(axis=1, how="all")
    df = df[~df.index.astype(str).str.startswith(("ENSG", "ENSP"))]
    df = df.select_dtypes(include=[np.number])
    df = df.loc[~df.index.duplicated(keep="first")]
    return df


def main():
    # ---------- 1. 药物反应 ----------
    ic50 = pd.read_excel(GDSC_IC50)
    ic50 = ic50.rename(columns={"CELL_LINE_NAME": "cell", "DRUG_NAME": "drug",
                                "LN_IC50": "lnIC50", "CANCER_TYPE": "cancer"})
    print(f"GDSC1 药物反应: {ic50.cell.nunique()} 细胞系 x {ic50.drug.nunique()} 药物")

    # ---------- 2. 表达矩阵 ----------
    gx = load_gdsc_expression()
    if gx is None:
        print("提示：rawdata/ 中未找到 GDSC 基因表达文件（文件名形如 Cell_line_RMA_proc_basalExp.txt），跳过。")
        return
    print(f"表达矩阵: {gx.shape[0]} 基因 x {gx.shape[1]} 细胞系")

    # 细胞系名称匹配（先精确，后忽略大小写）
    cells_ic50 = set(ic50.cell)
    common = [c for c in gx.columns if c in cells_ic50]
    if len(common) < 100:
        low = {c.lower(): c for c in gx.columns}
        common = [low[c.lower()] for c in cells_ic50 if c.lower() in low]
    print(f"细胞系匹配: {len(common)}")

    # ---------- 3. 细胞系风险分 ----------
    genes = pd.read_csv(f"{OUT}/model_genes.csv")
    coef = dict(zip(genes.gene, genes.coef))
    g4 = [g for g in coef if g in gx.index]
    assert len(g4) == len(coef), f"模型基因缺失: {set(coef) - set(g4)}"
    sub = gx.loc[g4, common].T          # 细胞系 x 4基因
    z = (sub - sub.mean()) / sub.std(ddof=0)
    risk = sum(z[g] * c for g, c in coef.items())
    risk = risk.rename("risk_score")
    risk.to_csv(f"{OUT}/gdsc_risk_score.csv")
    print(f"风险分范围: {risk.min():.2f} ~ {risk.max():.2f}")

    # ---------- 4. 全部药物 Spearman ----------
    piv = ic50.pivot_table(index="cell", columns="drug", values="lnIC50")
    piv = piv.reindex(common)
    rows = []
    for drug in piv.columns:
        y = piv[drug]
        ok = y.notna() & risk.notna()
        if ok.sum() < MIN_CELLS:
            continue
        rho, p = stats.spearmanr(risk[ok], y[ok])
        rows.append((drug, int(ok.sum()), rho, p))
    res = pd.DataFrame(rows, columns=["drug", "n_cells", "rho", "p"])
    res["fdr"] = bh_fdr(res.p)
    res = res.sort_values("rho")
    res.to_csv(f"{OUT}/drug_spearman.csv", index=False)
    sig = res[(res.fdr < 0.05) & (res.rho.abs() > 0.2)]
    print(f"显著药物 (FDR<0.05 & |rho|>0.2): {len(sig)} 个，敏感 {int((sig.rho<0).sum())} / 耐药 {int((sig.rho>0).sum())}")
    print(sig.sort_values('rho').to_string(index=False, max_rows=30))

    # ---------- 5. HCC 相关药物（全细胞系面板中的相关，供 Fig6C）----------
    hcc_cells = ic50[ic50.cancer.str.contains("liver|hepat", case=False, na=False)].cell.unique().tolist()
    hcc_cells = [c for c in hcc_cells if c in common]
    print(f"HCC 细胞系中具 RNA-seq 表达者: {len(hcc_cells)} -> {hcc_cells}")
    print("说明: CMP 合并 RNA-seq 仅覆盖少数肝癌细胞系，HCC 相关性改在全细胞系面板中评估。")

    # ---------- 6. 作图 ----------
    # Fig6A: 全景条形图（按 |rho| 取前 25，显著者着色，HCC 常用药标 *）
    top = res.reindex(res.rho.abs().sort_values(ascending=False).index).head(25)
    top = top.sort_values("rho")
    colors = [COL_RISK if r > 0 else COL_SAFE for r in top.rho]
    NAME_FIX = {"Venotoclax": "Venetoclax"}   # GDSC 原始拼写勘误
    labels = [NAME_FIX.get(d, d) + ("*" if d in HCC_DRUGS else "") for d in top.drug]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.barh(range(len(top)), top.rho, color=colors, alpha=0.9)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman rho (risk score vs ln IC50)")
    ax.set_title("Drug sensitivity landscape across GDSC1 cell lines", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COL_SAFE, label="Sensitive in high-risk (rho<0)"),
                       Patch(color=COL_RISK, label="Resistant in high-risk (rho>0)")],
              loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig6A_drug_landscape.pdf", dpi=FIG_DPI); plt.close(fig)

    # Fig6B: 代表药物散点（最敏感 + 最耐药）
    pick = [res.iloc[0], res.iloc[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (_, r) in zip(axes, enumerate(pick)):
        y = piv[r.drug]
        ok = y.notna() & risk.notna()
        x_, y_ = risk[ok], y[ok]
        ax.scatter(x_, y_, s=10, alpha=0.45, color="#7F7F7F", edgecolors="none")
        b, a = np.polyfit(x_, y_, 1)
        xs = np.linspace(x_.min(), x_.max(), 50)
        ax.plot(xs, a + b * xs, color=COL_RISK if r.rho > 0 else COL_SAFE, lw=2)
        tag = "Resistant" if r.rho > 0 else "Sensitive"
        ax.set_title(f"{r.drug} ({tag})\nrho={r.rho:.2f}, FDR={r.fdr:.1e}", fontsize=10)
        ax.set_xlabel("Risk score"); ax.set_ylabel("ln IC50")
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig6B_drug_scatter.pdf", dpi=FIG_DPI); plt.close(fig)

    # Fig6C: HCC 常用/在研药物的风险分相关性（全细胞系面板）
    panel = [d for d in HCC_DRUGS if d in piv.columns]
    pr = res.set_index("drug").reindex(panel).dropna(subset=["rho"]).sort_values("rho")
    if len(pr):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        colors = [COL_RISK if r > 0 else COL_SAFE for r in pr.rho]
        ax.barh(range(len(pr)), pr.rho, color=colors, alpha=0.9)
        ax.set_yticks(range(len(pr))); ax.set_yticklabels(pr.index, fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        for i, (rho, fdr) in enumerate(zip(pr.rho, pr.fdr)):
            star = "***" if fdr < 1e-3 else ("**" if fdr < 1e-2 else ("*" if fdr < 0.05 else "ns"))
            ax.text(rho + (0.008 if rho > 0 else -0.008), i, star, va="center",
                    ha="left" if rho > 0 else "right", fontsize=9)
        ax.set_xlabel("Spearman rho (risk score vs ln IC50)")
        ax.set_title("HCC-relevant compounds across GDSC1 cell lines\n(* FDR<0.05, ** FDR<0.01, *** FDR<0.001)", fontsize=10)
        ax.set_xlim(pr.rho.min() - 0.06, pr.rho.max() + 0.06)
        fig.tight_layout(); fig.savefig(f"{FIG}/Fig6C_hcc_drugs.pdf", dpi=FIG_DPI); plt.close(fig)

    print("06 完成")


if __name__ == "__main__":
    main()
