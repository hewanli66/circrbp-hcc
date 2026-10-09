# -*- coding: utf-8 -*-
"""
09b 比例风险假设检验 + 风险评分分布 + 中位随访（审稿意见 #4、#12）
内容：
  1) Cox 模型（连续风险评分）Schoenfeld 残差检验比例风险(PH)假设
  2) 风险评分分布：直方图 + 按生存结局分组箱线；极端值排查
     （最高分 1% 样本剔除后的 C-index 变化；99 分位截尾后的 C-index）
  3) TCGA 中位随访时间（reverse Kaplan-Meier）与删失分布
输出：results/rev4_ph_test.csv, results/rev4_score_dist_summary.csv,
      results/rev4_followup.csv, figs/FigS4_risk_distribution.pdf
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, COL_RISK, COL_SAFE, FIG_DPI
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import median_survival_times as _mst
def median_survival_time(km):
    v = _mst(km.survival_function_)
    return float(np.atleast_1d(np.asarray(v))[0])
from sksurv.metrics import concordance_index_censored
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"


def main():
    clin = pd.read_csv(f"{OUT}/tcga_clin.csv").set_index("sample")
    rs = pd.read_csv(f"{OUT}/risk_scores.csv").set_index("sample")["risk_score"]
    d = clin.copy(); d["risk"] = rs

    # ---------- 1) PH 假设 ----------
    cph = CoxPHFitter().fit(d[["risk", "OS.time", "OS"]], "OS.time", "OS")
    schoen = proportional_hazard_test(cph, d[["risk", "OS.time", "OS"]], time_transform="rank")
    ph = pd.DataFrame({"test": schoen.summary.index,
                       "test_statistic": schoen.summary["test_statistic"].values,
                       "p_value": schoen.summary["p"].values})
    ph.to_csv(f"{OUT}/rev4_ph_test.csv", index=False)
    print("Schoenfeld 检验（rank 时间变换）:")
    print(ph.to_string(index=False))
    print(f"HR(连续) = {np.exp(cph.params_['risk']):.2f} 每 1-SD, p = {cph.summary.loc['risk','p']:.2e}")

    # PH 不成立时的稳健补充：中位数二分后 5 年 RMST 差
    from lifelines.utils import restricted_mean_survival_time
    med = d.risk.median()
    km_h = KaplanMeierFitter().fit(d["OS.time"][d.risk > med], d.OS[d.risk > med])
    km_l = KaplanMeierFitter().fit(d["OS.time"][d.risk <= med], d.OS[d.risk <= med])
    rmst_h = restricted_mean_survival_time(km_h, 1826.25)
    rmst_l = restricted_mean_survival_time(km_l, 1826.25)
    pd.DataFrame({"group": ["high_risk", "low_risk"],
                  "RMST_5yr_days": [rmst_h, rmst_l]}).to_csv(f"{OUT}/rev4_rmst.csv", index=False)
    print(f"\n5 年 RMST: 高风险 {rmst_h:.0f} 天 vs 低风险 {rmst_l:.0f} 天 "
          f"(差 {rmst_l - rmst_h:.0f} 天)")

    # ---------- 2) 风险评分分布 ----------
    q99 = d.risk.quantile(0.99)
    top1 = d.risk > q99
    ci_all = concordance_index_censored(d.OS.astype(bool), d["OS.time"], d.risk)[0]
    ci_ex = concordance_index_censored(d.OS[~top1].astype(bool), d["OS.time"][~top1], d.risk[~top1])[0]
    rw = d.risk.clip(upper=q99)
    ci_win = concordance_index_censored(d.OS.astype(bool), d["OS.time"], rw)[0]
    summ = pd.DataFrame({
        "item": ["n", "mean", "sd", "median", "min", "max", "p99", "top1pct_n",
                 "c_index_all", "c_index_exclude_top1pct", "c_index_winsorized_p99"],
        "value": [len(d), d.risk.mean(), d.risk.std(), d.risk.median(), d.risk.min(),
                  d.risk.max(), q99, int(top1.sum()), ci_all, ci_ex, ci_win]})
    summ.to_csv(f"{OUT}/rev4_score_dist_summary.csv", index=False)
    print("\n风险评分分布:")
    print(summ.to_string(index=False))
    print(f"最高分 5 例: {d.risk.sort_values(ascending=False).head(5).round(3).to_dict()}")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].hist(d.risk, bins=40, color="#8FAADC", edgecolor="white")
    axes[0].axvline(d.risk.median(), ls="--", c="k", lw=1)
    axes[0].set_xlabel("Risk score"); axes[0].set_ylabel("Patients")
    axes[0].set_title("Risk score distribution (TCGA)", fontsize=10)
    n_alive, n_dead = int((d.OS == 0).sum()), int((d.OS == 1).sum())
    bp = axes[1].boxplot([d.risk[d.OS == 0], d.risk[d.OS == 1]],
                         labels=[f"Alive\n(n={n_alive})", f"Dead\n(n={n_dead})"],
                         patch_artist=True, widths=0.5,
                         medianprops=dict(color="black"))
    for p, c in zip(bp["boxes"], [COL_SAFE, COL_RISK]):
        p.set_facecolor(c); p.set_alpha(0.85)
    axes[1].set_ylabel("Risk score")
    axes[1].set_title("Risk score by vital status", fontsize=10)
    axes[2].hist(d.risk[~top1], bins=40, color="#8FAADC", edgecolor="white", label="other 99%")
    axes[2].hist(d.risk[top1], bins=8, color=COL_RISK, edgecolor="white", alpha=0.9, label="top 1%")
    axes[2].set_xlabel("Risk score"); axes[2].set_ylabel("Patients")
    axes[2].set_title(f"Extreme values (>{q99:.2f})", fontsize=10)
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/FigS4_risk_distribution.pdf", dpi=FIG_DPI); plt.close(fig)

    # ---------- 3) 中位随访（reverse KM）----------
    km = KaplanMeierFitter()
    km.fit(d["OS.time"], 1 - d.OS, label="censoring")
    fu = median_survival_time(km)
    fu_tab = pd.DataFrame({
        "item": ["median_followup_days", "median_followup_years", "n", "n_events",
                 "n_censored", "censoring_pct"],
        "value": [fu, fu / 365.25, len(d), int(d.OS.sum()), int((1 - d.OS).sum()),
                  (1 - d.OS).mean() * 100]})
    fu_tab.to_csv(f"{OUT}/rev4_followup.csv", index=False)
    print("\n随访概况:")
    print(fu_tab.to_string(index=False))
    print("\n09b 完成")


if __name__ == "__main__":
    main()
