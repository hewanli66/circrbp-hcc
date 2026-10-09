# -*- coding: utf-8 -*-
"""
09d 肝癌细胞系子集药敏分析（审稿意见 #6）
在 GDSC1 中限定肝癌来源细胞系（CANCER_TYPE 含 liver/hepatoma），
重算 4 基因风险评分与各化合物 LN_IC50 的 Spearman 相关，
作为"泛癌面板结论"的 HCC 特异性佐证（n 小，定位为支持性证据）。
关键药物类别：HDAC 抑制剂、BET 抑制剂、BCL-2 抑制剂、MEK 抑制剂、5-FU、CX-5461、索拉非尼等。
输出：results/rev6_drug_hcc_subset.csv, figs/FigS6_hcc_subset_drug.pdf
"""
import sys, os, glob
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, RAW, GDSC_IC50, COL_RISK, COL_SAFE, FIG_DPI
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

MIN_HCC = 8   # 每种药物在肝癌子集中最少细胞系数

KEY_CLASSES = {
    "HDAC inhibitor": ["Panobinostat", "Vorinostat", "Belinostat", "Tucidinostat", "Entinostat"],
    "BET inhibitor": ["JQ1", "OTX015", "I-BET-762", "Birabresib"],
    "BCL-2 inhibitor": ["ABT-263", "ABT-199", "Navitoclax", "Venetoclax", "Obatoclax", "Sabutoclax"],
    "MEK inhibitor": ["Trametinib", "Selumetinib", "PD-0325901", "CI-1040", "Refametinib", "Cobimetinib"],
    "Other key": ["5-Fluorouracil", "CX-5461", "Sorafenib", "Lenvatinib", "Regorafenib", "Epirubicin"],
}


def load_expression():
    """优先使用 GDSC 经典 RMA  basal 表达文件（覆盖全部 GDSC1 细胞系）；
    缺失时退回 Cell Model Passports 合并 RNA-seq（仅覆盖部分细胞系）。"""
    import glob as _glob
    rma = sorted(_glob.glob(os.path.join(RAW, "*[Rr][Mm][Aa]*")) +
                 _glob.glob(os.path.join(RAW, "*basalExp*")))
    rma = [p for p in rma if not p.endswith(".zip")]
    if rma:
        print(f"表达文件(RMA): {os.path.basename(rma[0])}")
        df = pd.read_csv(rma[0], sep="\t", index_col=0)
        df = df.dropna(axis=1, how="all")
        df = df[~df.index.astype(str).str.startswith(("ENSG", "ENSP"))]
        df = df.select_dtypes(include=[np.number])
        df = df.loc[~df.index.duplicated(keep="first")]
        return df
    path = os.path.join(RAW, "rnaseq_merged_rsem_fpkm_20260323.csv")
    print(f"表达文件(CMP RNA-seq): {os.path.basename(path)}（注意：肝癌细胞系覆盖极少）")
    df = pd.read_csv(path, index_col=0)
    if "model_name" in df.index:
        names = df.loc["model_name"].dropna()
        df = df.drop(index=["model_name", "data_source", "gene_symbol"])
        df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
        df.columns = [names.get(c, c) for c in df.columns]
    df = df[~df.index.duplicated(keep="first")]
    return df.apply(pd.to_numeric, errors="coerce")


def main():
    ic50 = pd.read_excel(GDSC_IC50)
    ic50 = ic50.rename(columns={"CELL_LINE_NAME": "cell", "DRUG_NAME": "drug",
                                "LN_IC50": "lnIC50", "CANCER_TYPE": "cancer"})
    hcc_cells = ic50[ic50.cancer.str.contains("liver|hepat", case=False, na=False)].cell.unique()
    print(f"GDSC1 中肝癌相关细胞系（按癌型标注）: {len(hcc_cells)} -> {sorted(hcc_cells)}")

    gx = load_expression()
    # 名称规范化匹配：大写 + 仅保留字母数字（兼容 HuH-7/HUH7/Hep3B2-1-7 等写法差异）
    def norm(s):
        return "".join(ch for ch in str(s).upper() if ch.isalnum())
    gx_norm = {norm(c): c for c in gx.columns}
    common = [gx_norm[norm(c)] for c in hcc_cells if norm(c) in gx_norm]
    print(f"其中有 CMP RNA-seq 表达者: {len(common)} -> {sorted(common)}")

    coef = pd.read_csv(f"{OUT}/model_genes.csv").set_index("gene")["coef"]
    sub = gx.loc[[g for g in coef.index if g in gx.index], common].T.astype(float)
    sub = sub.dropna(axis=1, how="all")
    z = (sub - sub.mean()) / sub.std(ddof=0)
    risk = (z * coef[sub.columns]).sum(axis=1)
    print(f"肝癌子集风险分: {len(risk)} 株, 范围 {risk.min():.2f}~{risk.max():.2f}")

    piv = ic50.pivot_table(index="cell", columns="drug", values="lnIC50")
    piv.index = [norm(c) for c in piv.index]
    piv = piv[~piv.index.duplicated()].reindex([norm(c) for c in common])
    rows = []
    for drug in piv.columns:
        y = piv[drug]
        ok = y.notna() & risk.notna()
        if ok.sum() < MIN_HCC:
            continue
        rho, p = stats.spearmanr(risk[ok], y[ok])
        rows.append((drug, int(ok.sum()), rho, p))
    res = pd.DataFrame(rows, columns=["drug", "n_hcc_lines", "rho", "p"]).sort_values("rho")
    res.to_csv(f"{OUT}/rev6_drug_hcc_subset.csv", index=False)
    print(f"\n肝癌子集中可评估药物: {len(res)} 种")

    # 与既往全面板结果对照（关键药）
    full = pd.read_csv(f"{OUT}/drug_spearman.csv").set_index("drug")
    key_rows = []
    for cls, drugs in KEY_CLASSES.items():
        for dname in drugs:
            hit = res[res.drug.str.contains(dname, case=False, regex=False)]
            if len(hit):
                r = hit.iloc[0]
                full_rho = full.rho.get(r.drug, np.nan)
                full_fdr = full.fdr.get(r.drug, np.nan)
                key_rows.append((cls, r.drug, r.n_hcc_lines, r.rho, r.p, full_rho, full_fdr))
    key_tab = pd.DataFrame(key_rows, columns=["class", "drug", "n_hcc", "rho_hcc", "p_hcc",
                                              "rho_pancan", "fdr_pancan"])
    key_tab.to_csv(f"{OUT}/rev6_key_drugs.csv", index=False)
    print("\n关键药物类别（肝癌子集 vs 泛癌面板）:")
    print(key_tab.to_string(index=False))

    # 图：关键药物 rho 对比
    if len(key_tab):
        fig, ax = plt.subplots(figsize=(8.5, max(4, len(key_tab) * 0.32)))
        kt = key_tab.sort_values("rho_hcc")
        colors = [COL_SAFE if r < 0 else COL_RISK for r in kt.rho_hcc]
        ax.barh(range(len(kt)), kt.rho_hcc, color=colors, alpha=0.9)
        ax.set_yticks(range(len(kt)))
        ax.set_yticklabels([f"{c} | {d} (n={n})" for c, d, n in zip(kt["class"], kt.drug, kt.n_hcc)],
                           fontsize=8)
        ax.axvline(0, c="k", lw=0.8)
        ax.axvline(0.2, ls=":", c="grey", lw=0.8); ax.axvline(-0.2, ls=":", c="grey", lw=0.8)
        ax.set_xlabel("Spearman rho (risk score vs ln IC50, HCC cell lines)")
        ax.set_title(f"HCC-lineage subset (n={len(common)} cell lines)\nblue=sensitive in high-risk, red=resistant", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{FIG}/FigS6_hcc_subset_drug.pdf", dpi=FIG_DPI); plt.close(fig)
    print("\n09d 完成")


if __name__ == "__main__":
    main()
