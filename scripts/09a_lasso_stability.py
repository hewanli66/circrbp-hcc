# -*- coding: utf-8 -*-
"""
09a LASSO-Cox 模型稳定性分析（审稿意见 #3）
内容：
  1) 明确原模型的 lambda 选择准则：在 LASSO 正则化路径上取【最先出现（最简约）的 4 基因解】
     —— 即在最强正则化（最大 lambda）下即已入选的 4 个基因。
  2) bootstrap 200 次：固定原 lambda，记录 60 个候选基因的非零入选频率；对 4 个模型基因
     报告 bootstrap 系数的 2.5/50/97.5 分位数。
  3) 3 基因模型（剔除 NOVA1，沿用原系数）vs 4 基因模型：
     TCGA C-index / 1年AUC / 连续Cox HR / KM p；GSE14520(OS)、GSE76427(OS) KM p；
     5 折 CV C-index（各自在折内重拟合）。
输出：results/rev3_stability_lambda.csv   固定lambda下bootstrap入选频率
      results/rev3_stability_coef.csv     模型基因bootstrap系数分位数
      results/rev3_model_compare.csv      3基因 vs 4基因全面对比
      figs/FigS3_stability.pdf            bootstrap系数箱线图 + 模型对比条形图
"""
import sys, os, warnings
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, COL_RISK, COL_SAFE, FIG_DPI
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

B_BOOT = 200
SEED = 20261009


def load_tcga():
    expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
    clin = pd.read_csv(f"{OUT}/tcga_clin.csv").set_index("sample")
    cand = pd.read_csv(f"{OUT}/univ_cox.csv").gene.tolist()
    sub = expr.loc[cand, clin.index].T
    sub_z = (sub - sub.mean()) / sub.std(ddof=0)
    y = Surv.from_arrays(clin["OS"].astype(bool).values, clin["OS.time"].values)
    return sub_z, y, clin


def geo_risk(ge, gc, coef: dict, event_col, time_col):
    """与 03_model.py 相同的跨队列风险分计算（各队列内部 z 分数）"""
    ge = ge[~ge.index.duplicated()]
    common = [g for g in coef if g in ge.index]
    gd = ge.loc[common].T
    gd = gd.loc[:, gd.std(ddof=0) > 0]
    z = (gd - gd.mean()) / gd.std(ddof=0)
    grs = (z * pd.Series({g: coef[g] for g in gd.columns})).sum(axis=1)
    gdat = pd.concat([grs.rename("risk"), gc.set_index("sample")], axis=1)
    gdat = gdat.dropna(subset=[event_col, time_col, "risk"])
    gdat = gdat[gdat[time_col] >= 30]
    return gdat


def cohort_stats(d, score_col="risk", time_col="OS.time", event_col="OS"):
    """返回 (C-index, 1年AUC或None, 连续HR, KM logrank p)"""
    ci = concordance_index_censored(d[event_col].astype(bool).values, d[time_col].values,
                                    d[score_col].values)[0]
    cph = CoxPHFitter().fit(d[[score_col, time_col, event_col]], time_col, event_col)
    hr, hr_p = np.exp(cph.params_[score_col]), cph.summary.loc[score_col, "p"]
    auc1 = None
    status = ((d[event_col] == 1) & (d[time_col] <= 365)).astype(int)
    if status.nunique() == 2:
        fpr, tpr, _ = roc_curve(status, d[score_col])
        auc1 = auc(fpr, tpr)
    med = d[score_col].median()
    hi, lo = d[d[score_col] > med], d[d[score_col] <= med]
    p = logrank_test(hi[time_col], lo[time_col], hi[event_col], lo[event_col]).p_value
    return ci, auc1, hr, hr_p, p


def main():
    sub_z, y, clin = load_tcga()
    n, p = sub_z.shape

    # ---------- 1) 原始路径：找最简约的 4 基因解 ----------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cox = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=60)
        cox.fit(sub_z.values, y)
    lam_orig, coef4 = None, None
    for li, lam in enumerate(cox.alphas_):
        genes = [g for g, b in zip(sub_z.columns, cox.coef_[:, li] != 0) if b]
        if len(genes) == 4:
            lam_orig = float(lam)
            coef4 = pd.Series({g: float(cox.coef_[list(sub_z.columns).index(g), li]) for g in genes})
            break
    print(f"lambda 选择准则: 正则化路径上最简约(最先出现)的 4 基因解, lambda = {lam_orig:.4f}")
    print(f"原模型系数: {coef4.to_dict()}")

    # ---------- 2) bootstrap 稳定性 ----------
    rng = np.random.RandomState(SEED)
    freq = pd.DataFrame(0, index=sub_z.columns, columns=["selected"])
    coef_boot = {g: [] for g in coef4.index}
    nonzero_count = []
    for b in range(B_BOOT):
        idx = rng.randint(0, n, n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mb = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[lam_orig]).fit(sub_z.values[idx], y[idx])
            nz = sub_z.columns[np.abs(mb.coef_[:, 0]) > 1e-8]
            nonzero_count.append(len(nz))
            freq.loc[nz, "selected"] += 1
            for g in coef4.index:
                j = list(sub_z.columns).index(g)
                if abs(mb.coef_[j, 0]) > 1e-8:
                    coef_boot[g].append(float(mb.coef_[j, 0]))
        except Exception:
            nonzero_count.append(np.nan)
    freq["freq_%"] = freq["selected"] / B_BOOT * 100
    freq = freq.sort_values("freq_%", ascending=False)
    freq.to_csv(f"{OUT}/rev3_stability_lambda.csv")

    rows = []
    for g, cs in coef_boot.items():
        cs = np.array(cs)
        rows.append((g, len(cs), np.percentile(cs, 2.5), np.percentile(cs, 50),
                     np.percentile(cs, 97.5), np.mean(cs > 1e-8)))
    coef_q = pd.DataFrame(rows, columns=["gene", "n_selected", "coef_p2.5", "coef_p50",
                                         "coef_p97.5", "prop_positive"])
    coef_q.to_csv(f"{OUT}/rev3_stability_coef.csv", index=False)
    print("\nbootstrap 入选频率 top10:")
    print(freq.head(10).to_string())
    print("\n模型基因 bootstrap 系数分位数:")
    print(coef_q.to_string(index=False))
    print(f"\nbootstrap 非零基因数: 中位 {np.nanmedian(nonzero_count):.0f}, "
          f"恰好4基因 {(np.array(nonzero_count) == 4).mean() * 100:.1f}%")

    # ---------- 3) 3 基因 vs 4 基因 ----------
    coef3 = coef4.drop("NOVA1")
    d = clin.copy()
    d["risk4"] = (sub_z * coef4).sum(axis=1)
    d["risk3"] = (sub_z * coef3).sum(axis=1)

    ge14 = pd.read_csv(f"{OUT}/gse14520_expr_symbol.csv", index_col=0)
    gc14 = pd.read_csv(f"{OUT}/gse14520_clin_raw.csv")
    ge76 = pd.read_csv(f"{OUT}/gse76427_expr_symbol.csv", index_col=0)
    gc76 = pd.read_csv(f"{OUT}/gse76427_clin_raw.csv")
    g14_4 = geo_risk(ge14, gc14, coef4.to_dict(), "OS", "OS.time")
    g14_3 = geo_risk(ge14, gc14, coef3.to_dict(), "OS", "OS.time")
    g76_4 = geo_risk(ge76, gc76, coef4.to_dict(), "OS", "OS.time")
    g76_3 = geo_risk(ge76, gc76, coef3.to_dict(), "OS", "OS.time")

    res = []
    for name, dd, sc in [("TCGA", d, "risk4"), ("TCGA", d, "risk3"),
                         ("GSE14520_OS", g14_4, "risk"), ("GSE14520_OS", g14_3, "risk"),
                         ("GSE76427_OS", g76_4, "risk"), ("GSE76427_OS", g76_3, "risk")]:
        ci, a1, hr, hrp, p = cohort_stats(dd, sc)
        res.append((name, "4-gene" if sc == "risk4" and name == "TCGA" or sc == "risk" and dd is g14_4 or sc == "risk" and dd is g76_4 else "3-gene",
                    len(dd), int(dd[dd.columns[dd.columns.get_loc("OS")]].sum()) if "OS" in dd.columns else int(dd.OS.sum()),
                    ci, a1, hr, hrp, p))
    cmp_df = pd.DataFrame(res, columns=["cohort", "model", "n", "events", "c_index", "auc_1yr",
                                        "HR_cont", "HR_p", "logrank_p"])
    cmp_df.to_csv(f"{OUT}/rev3_model_compare.csv", index=False)
    print("\n3 基因 vs 4 基因对比:")
    print(cmp_df.to_string(index=False))

    # 5 折 CV C-index（折内重拟合）
    kf = KFold(5, shuffle=True, random_state=42)
    cv4, cv3 = [], []
    for tr, va in kf.split(sub_z.values):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m4 = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[lam_orig]).fit(sub_z.values[tr][:, [list(sub_z.columns).index(g) for g in coef4.index]], y[tr])
            m3 = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[lam_orig]).fit(sub_z.values[tr][:, [list(sub_z.columns).index(g) for g in coef3.index]], y[tr])
        j4 = [list(sub_z.columns).index(g) for g in coef4.index]
        j3 = [list(sub_z.columns).index(g) for g in coef3.index]
        cv4.append(concordance_index_censored(y[va]["event"], y[va]["time"],
                    sub_z.values[va][:, j4] @ m4.coef_[:, 0])[0])
        cv3.append(concordance_index_censored(y[va]["event"], y[va]["time"],
                    sub_z.values[va][:, j3] @ m3.coef_[:, 0])[0])
    print(f"\n5折CV C-index: 4基因={np.mean(cv4):.3f}  3基因={np.mean(cv3):.3f}")
    with open(f"{OUT}/rev3_cv_cindex.txt", "w") as f:
        f.write(f"5-fold CV C-index, 4-gene={np.mean(cv4):.4f} (+-{np.std(cv4):.4f}), "
                f"3-gene={np.mean(cv3):.4f} (+-{np.std(cv3):.4f})\n")

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bp_data = [coef_boot[g] for g in coef4.index]
    axes[0].boxplot(bp_data, labels=coef4.index, showfliers=False,
                    medianprops=dict(color="black"))
    for i, g in enumerate(coef4.index, 1):
        xs = np.random.normal(i, 0.04, len(coef_boot[g]))
        axes[0].scatter(xs, coef_boot[g], s=6, alpha=0.25, color="#7F7F7F", edgecolors="none")
        axes[0].hlines(coef4[g], i - 0.3, i + 0.3, color=COL_RISK, lw=2)
    axes[0].set_ylabel("LASSO coefficient (bootstrap)")
    axes[0].set_title(f"Bootstrap coefficient stability (B={B_BOOT})\nred line = original fit", fontsize=10)

    t = cmp_df[cmp_df.cohort == "TCGA"].set_index("model")
    metrics = ["c_index", "auc_1yr"]
    x = np.arange(len(metrics))
    axes[1].bar(x - 0.18, [t.loc["4-gene", m] for m in metrics], 0.36, label="4-gene", color=COL_RISK)
    axes[1].bar(x + 0.18, [t.loc["3-gene", m] for m in metrics], 0.36, label="3-gene (drop NOVA1)", color=COL_SAFE)
    axes[1].set_xticks(x); axes[1].set_xticklabels(["C-index (TCGA)", "1-yr AUC (TCGA)"])
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("3-gene vs 4-gene discrimination", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/FigS3_stability.pdf", dpi=FIG_DPI); plt.close(fig)
    print("\n09a 完成")


if __name__ == "__main__":
    main()
