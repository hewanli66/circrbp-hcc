# -*- coding: utf-8 -*-
"""
09f 基准比较：4基因签名 vs 增殖元基因（审稿意见 #8）
以 MSigDB HALLMARK_G2M_CHECKPOINT 基因集的均值 z 分数作为增殖元基因评分，
与 4 基因风险评分比较：C-index、1年AUC、KM p（TCGA、GSE14520、GSE76427），
二者 Spearman 相关；联合模型(评分+增殖)C-index 与 Cox 多因素检验评估增量价值。
输出：results/rev8_benchmark.csv, figs/FigS8_benchmark.pdf
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, GENESET, COL_RISK, COL_SAFE, FIG_DPI
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_curve, auc
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"


def load_g2m_genes():
    with open(os.path.join(GENESET, "h.all.v7.5.1.symbols.gmt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("HALLMARK_G2M_CHECKPOINT"):
                return set(line.rstrip("\n").split("\t")[2:])
    raise RuntimeError("GMT 中未找到 HALLMARK_G2M_CHECKPOINT")


def metagene(ge: pd.DataFrame, genes: set):
    """均值 z 分数增殖元基因评分（行=基因, 列=样本）"""
    g2 = [g for g in genes if g in ge.index]
    sub = ge.loc[g2].astype(float)
    z = (sub.T - sub.T.mean()) / sub.T.std(ddof=0)
    return z.mean(axis=1), len(g2)


def geo_frame(ge, gc, coef, event_col, time_col):
    ge = ge[~ge.index.duplicated()]
    common = [g for g in coef if g in ge.index]
    gd = ge.loc[common].T
    gd = gd.loc[:, gd.std(ddof=0) > 0]
    z = (gd - gd.mean()) / gd.std(ddof=0)
    grs = (z * pd.Series({g: coef[g] for g in gd.columns})).sum(axis=1)
    gdat = pd.concat([grs.rename("risk"), gc.set_index("sample")], axis=1)
    gdat = gdat.dropna(subset=[event_col, time_col, "risk"])
    return gdat[gdat[time_col] >= 30]


def ev(d, score, time_col="OS.time", event_col="OS"):
    ci = concordance_index_censored(d[event_col].astype(bool), d[time_col].values, d[score].values)[0]
    st = ((d[event_col] == 1) & (d[time_col] <= 365)).astype(int)
    a1 = auc(*roc_curve(st, d[score])[0:2]) if st.nunique() == 2 else np.nan
    med = d[score].median()
    hi, lo = d[d[score] > med], d[d[score] <= med]
    p = logrank_test(hi[time_col], lo[time_col], hi[event_col], lo[event_col]).p_value
    return ci, a1, p


def main():
    g2m = load_g2m_genes()
    coef = pd.read_csv(f"{OUT}/model_genes.csv").set_index("gene")["coef"].to_dict()

    # ---------- TCGA ----------
    expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
    clin = pd.read_csv(f"{OUT}/tcga_clin.csv").set_index("sample")
    risk = pd.read_csv(f"{OUT}/risk_scores.csv").set_index("sample")["risk_score"]
    d = clin.copy(); d["risk"] = risk
    mg, n_g2m = metagene(expr, g2m)
    d["prolif"] = mg.loc[d.index]
    rho, prho = stats.spearmanr(d.risk, d.prolif)
    print(f"G2M 基因集: {n_g2m} 个基因映射成功")
    print(f"风险评分 vs 增殖元基因 Spearman rho = {rho:.3f} (p={prho:.1e})")

    ci_r, a1_r, p_r = ev(d, "risk")
    ci_m, a1_m, p_m = ev(d, "prolif")
    ci_joint = concordance_index_censored(d.OS.astype(bool), d["OS.time"],
                                          CoxPHFitter().fit(d[["risk", "prolif", "OS.time", "OS"]],
                                                            "OS.time", "OS").predict_partial_hazard(d[["risk", "prolif"]]).values.flatten())[0]
    cph2 = CoxPHFitter().fit(d[["risk", "prolif", "OS.time", "OS"]], "OS.time", "OS")
    tab_rows = [("TCGA", len(d), int(d.OS.sum()), "4-gene signature", ci_r, a1_r, p_r),
                ("TCGA", len(d), int(d.OS.sum()), "G2M metagene", ci_m, a1_m, p_m)]

    # ---------- GEO 队列 ----------
    for name, gef, gcf in [("GSE14520_OS", "gse14520_expr_symbol.csv", "gse14520_clin_raw.csv"),
                           ("GSE76427_OS", "gse76427_expr_symbol.csv", "gse76427_clin_raw.csv")]:
        ge = pd.read_csv(f"{OUT}/{gef}", index_col=0)
        gc = pd.read_csv(f"{OUT}/{gcf}")
        g = geo_frame(ge, gc, coef, "OS", "OS.time")
        mg2, _ = metagene(ge, g2m)
        g["prolif"] = mg2.loc[g.index]
        ci_r2, a1_r2, p_r2 = ev(g, "risk")
        ci_m2, a1_m2, p_m2 = ev(g, "prolif")
        tab_rows.append((name, len(g), int(g.OS.sum()), "4-gene signature", ci_r2, a1_r2, p_r2))
        tab_rows.append((name, len(g), int(g.OS.sum()), "G2M metagene", ci_m2, a1_m2, p_m2))

    tab = pd.DataFrame(tab_rows, columns=["cohort", "n", "events", "model", "c_index", "auc_1yr", "logrank_p"])
    tab.to_csv(f"{OUT}/rev8_benchmark.csv", index=False)
    print("\n基准比较:")
    print(tab.to_string(index=False))
    print(f"\nTCGA 联合模型 C-index = {ci_joint:.3f}")
    print("\nTCGA Cox 多因素 (risk + prolif):")
    print(cph2.summary[["coef", "exp(coef)", "p"]].to_string())

    with open(f"{OUT}/rev8_summary.txt", "w") as f:
        f.write(f"Spearman(risk, prolif) = {rho:.4f}, p = {prho:.3e}\n")
        f.write(f"TCGA joint C-index = {ci_joint:.4f}\n")
        f.write(cph2.summary[["coef", "exp(coef)", "p"]].to_string())

    # 图
    fig, ax = plt.subplots(figsize=(7, 4))
    cohorts = ["TCGA", "GSE14520_OS", "GSE76427_OS"]
    x = np.arange(len(cohorts)); w = 0.36
    tr = tab[tab.model == "4-gene signature"].set_index("cohort")
    tm = tab[tab.model == "G2M metagene"].set_index("cohort")
    ax.bar(x - w / 2, [tr.loc[c, "c_index"] for c in cohorts], w, label="4-gene circRBP signature", color=COL_RISK)
    ax.bar(x + w / 2, [tm.loc[c, "c_index"] for c in cohorts], w, label="G2M proliferation metagene", color=COL_SAFE)
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(cohorts)
    ax.set_ylabel("C-index (OS)")
    ax.set_ylim(0.45, 0.75)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Benchmark against a proliferation metagene", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/FigS8_benchmark.pdf", dpi=FIG_DPI); plt.close(fig)
    print("\n09f 完成")


if __name__ == "__main__":
    main()
