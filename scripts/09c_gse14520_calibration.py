# -*- coding: utf-8 -*-
"""
09c 外部队列校准曲线：GSE14520 OS（审稿意见 #5、#12）
方法：
  1) TCGA 内拟合 Cox 模型（协变量=4基因连续风险评分），取基线生存函数 S0(t)
  2) GSE14520 患者按同一公式得到预测生存率 S_i(t) = S0(t)^exp(beta*(rs_i - mean_rs_TCGA))
  3) 按预测 3 年风险五分位分组，比较各组实际（Kaplan-Meier）与平均预测生存率
输出：results/rev5_gse14520_calibration.csv, figs/FigS5_calibration_gse14520.pdf
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, FIG_DPI, COL_GROUP
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

YEARS = [1, 3, 5]


def geo_risk(ge, gc, coef: dict, event_col, time_col):
    ge = ge[~ge.index.duplicated()]
    common = [g for g in coef if g in ge.index]
    gd = ge.loc[common].T
    gd = gd.loc[:, gd.std(ddof=0) > 0]
    z = (gd - gd.mean()) / gd.std(ddof=0)
    grs = (z * pd.Series({g: coef[g] for g in gd.columns})).sum(axis=1)
    gdat = pd.concat([grs.rename("risk"), gc.set_index("sample")], axis=1)
    gdat = gdat.dropna(subset=[event_col, time_col, "risk"])
    return gdat[gdat[time_col] >= 30]


def main():
    coef = pd.read_csv(f"{OUT}/model_genes.csv").set_index("gene")["coef"].to_dict()
    clin = pd.read_csv(f"{OUT}/tcga_clin.csv").set_index("sample")
    rs = pd.read_csv(f"{OUT}/risk_scores.csv").set_index("sample")["risk_score"]
    d = clin.copy(); d["risk"] = rs

    # TCGA Cox 拟合 + 基线生存
    cph = CoxPHFitter().fit(d[["risk", "OS.time", "OS"]], "OS.time", "OS")
    beta = cph.params_["risk"]
    mean_rs_tcga = d.risk.mean()
    s0 = cph.baseline_survival_
    s0t = {yr: float(s0.iloc[np.searchsorted(s0.index.values, yr * 365.25)].iloc[0]) for yr in YEARS}
    print("TCGA 基线生存率 S0(1/3/5yr):", {k: round(v, 3) for k, v in s0t.items()})

    # GSE14520 风险分与预测生存率
    ge14 = pd.read_csv(f"{OUT}/gse14520_expr_symbol.csv", index_col=0)
    gc14 = pd.read_csv(f"{OUT}/gse14520_clin_raw.csv")
    g = geo_risk(ge14, gc14, coef, "OS", "OS.time")
    for yr in YEARS:
        g[f"pred_{yr}yr"] = s0t[yr] ** np.exp(beta * (g.risk - mean_rs_tcga))
    g["risk_group"] = pd.qcut(g[f"pred_3yr"], 5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"])

    # 各组实际 vs 预测
    rows = []
    for grp, sub in g.groupby("risk_group", observed=True):
        km = KaplanMeierFitter().fit(sub["OS.time"], sub.OS)
        row = {"group": grp, "n": len(sub), "events": int(sub.OS.sum())}
        for yr in YEARS:
            t = yr * 365.25
            row[f"obs_{yr}yr"] = float(km.survival_function_at_times(t).iloc[0]) if t <= sub["OS.time"].max() else np.nan
            row[f"pred_{yr}yr"] = sub[f"pred_{yr}yr"].mean()
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/rev5_gse14520_calibration.csv", index=False)
    print("\nGSE14520 校准表（五分位）:")
    print(tab.to_string(index=False))

    # 作图
    fig, ax = plt.subplots(figsize=(5, 4.8))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="Ideal")
    markers = {1: "o", 3: "s", 5: "^"}
    for yr in YEARS:
        ok = tab[f"obs_{yr}yr"].notna()
        ax.scatter(tab.loc[ok, f"pred_{yr}yr"], tab.loc[ok, f"obs_{yr}yr"],
                   marker=markers[yr], s=55, color=COL_GROUP[0] if yr == 1 else (COL_GROUP[1] if yr == 3 else "#7A5195"),
                   label=f"{yr}-year", zorder=3, edgecolors="white", lw=0.6)
    ax.set_xlabel("Predicted survival (TCGA-fitted model)")
    ax.set_ylabel("Observed survival (GSE14520, KM)")
    ax.set_title("Calibration in external cohort GSE14520\n(quintiles of predicted 3-year risk)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/FigS5_calibration_gse14520.pdf", dpi=FIG_DPI); plt.close(fig)
    print("\n09c 完成")


if __name__ == "__main__":
    main()
