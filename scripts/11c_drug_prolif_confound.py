# -*- coding: utf-8 -*-
"""
11c 药敏增殖混杂控制（二审 Major 3）
在 399 个 GDSC1 细胞系上：
  1) G2M metagene 与 4 基因风险评分的 Spearman 相关
  2) 关键药物 (panobinostat, I-BET-762, venetoclax, trametinib, refametinib, PD0325901,
     selumetinib, 5-Fluorouracil, CX-5461) 的 rho：原始 vs 偏 Spearman（控制 metagene）
  3) 分层分析：按 metagene 三分位分层后，层内风险分-药物相关
输出：results/rev10_drug_prolif_confound.csv, figs/FigS10b_drug_prolif.pdf
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, RAW, GDSC_IC50, FIG_DPI
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEY_DRUGS = ["Panobinostat", "I-BET-762", "Venotoclax", "Trametinib", "Refametinib",
             "PD0325901", "Selumetinib", "5-Fluorouracil", "CX-5461"]


def load_gx():
    import glob
    cands = []
    for pat in ("*RMA*", "*rma*", "*basalExp*", "*fpkm*", "*FPKM*"):
        cands += glob.glob(os.path.join(RAW, pat))
    cands = [c for c in cands if not c.endswith(".zip")
             and not os.path.basename(c).startswith(("TCGA", "GSE", "TEST", "Li2017"))]
    path = sorted(cands)[0]
    print("表达文件:", os.path.basename(path))
    sep = "\t" if path.endswith((".txt", ".tsv")) else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    if "model_name" in df.index:
        names = df.loc["model_name"].dropna()
        df = df.drop(index=["model_name", "data_source", "gene_symbol"])
        df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
        df.columns = [names.get(c, c) for c in df.columns]
        df = df[~df.index.duplicated(keep="first")]
        df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    return df


def partial_spearman(x, y, cov):
    """秩偏相关：对秩做残差化后 Pearson"""
    rx, ry, rc = stats.rankdata(x), stats.rankdata(y), stats.rankdata(cov)
    A = np.column_stack([np.ones(len(rc)), rc])
    bx = np.linalg.lstsq(A, rx, rcond=None)[0]
    by = np.linalg.lstsq(A, ry, rcond=None)[0]
    r_x, r_y = rx - A @ bx, ry - A @ by
    pr, pp = stats.pearsonr(r_x, r_y)
    n = len(x)
    # 近似 p（自由度减 1）
    t = pr * np.sqrt((n - 3) / max(1e-12, 1 - pr ** 2))
    pp = 2 * stats.t.sf(abs(t), n - 3)
    return pr, pp


def main():
    risk = pd.read_csv(f"{OUT}/gdsc_risk_score.csv", index_col=0).iloc[:, 0].rename("risk")
    gx = load_gx()
    common = [c for c in risk.index if c in gx.columns]
    print(f"共同细胞系: {len(common)}")
    gx = gx[common]

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "genesets", "h.all.v7.5.1.symbols.gmt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("HALLMARK_G2M_CHECKPOINT"):
                g2m = set(line.rstrip("\n").split("\t")[2:])
    g2 = [g for g in g2m if g in gx.index]
    sub = gx.loc[g2].astype(float)
    z = (sub.T - sub.T.mean()) / sub.T.std(ddof=0)
    prolif = z.mean(axis=1).rename("prolif")
    print(f"G2M 基因映射 {len(g2)}; metagene 范围 {prolif.min():.2f}~{prolif.max():.2f}")

    rho_all, p_all = stats.spearmanr(risk.loc[common], prolif.loc[common])
    print(f"细胞系中 Spearman(risk, G2M metagene) = {rho_all:.3f}, p={p_all:.1e}")

    ic50 = pd.read_excel(GDSC_IC50).rename(columns={"CELL_LINE_NAME": "cell", "DRUG_NAME": "drug", "LN_IC50": "lnIC50"})
    piv = ic50.pivot_table(index="cell", columns="drug", values="lnIC50").reindex(common)

    rows = []
    for drug in KEY_DRUGS:
        y = piv[drug] if drug in piv.columns else None
        if y is None:
            # 大小写/拼写兜底
            cand = [c for c in piv.columns if c.lower() == drug.lower()]
            if not cand:
                print("缺药:", drug); continue
            y = piv[cand[0]]; drug = cand[0]
        ok = y.notna() & risk.notna() & prolif.notna()
        x_, y_, c_ = risk[ok], y[ok], prolif[ok]
        r0, p0 = stats.spearmanr(x_, y_)
        rp, pp = partial_spearman(x_.values, y_.values, c_.values)
        rows.append(dict(drug=drug, n=int(ok.sum()), rho_raw=r0, p_raw=p0,
                         rho_partial=rp, p_partial=pp))
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/rev10_drug_prolif_confound.csv", index=False)
    print(res.to_string(index=False))

    # 分层：metagene 三分位，层内相关
    tert = pd.qcut(prolif.loc[common], 3, labels=["low", "mid", "high"])
    strat_rows = []
    for drug in KEY_DRUGS:
        for lvl in ["low", "mid", "high"]:
            cells = tert[tert == lvl].index
            y = piv.loc[cells, drug] if drug in piv.columns else None
            if y is None:
                cand = [c for c in piv.columns if c.lower() == drug.lower()]
                if not cand: continue
                y = piv.loc[cells, cand[0]]
            ok = y.notna() & risk.notna()
            if ok.sum() >= 20 and risk[ok].std() > 0 and y[ok].std() > 0:
                r, p = stats.spearmanr(risk[ok], y[ok])
                strat_rows.append((drug, lvl, int(ok.sum()), r, p))
    strat = pd.DataFrame(strat_rows, columns=["drug", "prolif_tertile", "n", "rho_within", "p_within"])
    strat.to_csv(f"{OUT}/rev10_drug_stratified.csv", index=False)

    # 图：raw vs partial
    fig, ax = plt.subplots(figsize=(7, 4))
    xpos = np.arange(len(res))
    ax.bar(xpos - 0.2, res.rho_raw, 0.38, label="Raw Spearman", color="#2E5A88")
    ax.bar(xpos + 0.2, res.rho_partial, 0.38, label="Partial (adjusted for G2-M metagene)", color="#C0504D")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels(res.drug, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("rho with LN_IC50 (risk score)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Drug correlations before/after proliferation adjustment\nSpearman(risk, G2M) = {rho_all:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/FigS10b_drug_prolif.pdf", dpi=FIG_DPI); plt.close(fig)
    print("\n11c 完成")


if __name__ == "__main__":
    main()
