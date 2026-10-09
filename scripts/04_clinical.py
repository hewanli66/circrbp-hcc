# -*- coding: utf-8 -*-
"""
04 独立预后因素分析 + 列线图（nomogram）
输出：
  figs/Fig4A_forest_uni.pdf / Fig4B_forest_multi.pdf   单因素/多因素 Cox 森林图
  figs/Fig4C_nomogram.pdf                              列线图
  figs/Fig4D_calibration.pdf                           校准曲线（1/3/5年）
  figs/Fig4E_pca.pdf                                   高低风险组 PCA 分型
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG
from lifelines import CoxPHFitter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"


def forest(cph, tag):
    s = cph.summary
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(s) + 1.5))
    y = range(len(s))[::-1]
    ax.errorbar(s["exp(coef)"], y, xerr=[s["exp(coef)"] - s["exp(coef) lower 95%"],
                s["exp(coef) upper 95%"] - s["exp(coef)"]], fmt="s", c="k", ecolor="grey", capsize=2)
    ax.axvline(1, ls="--", c="r", lw=.8)
    ax.set_yticks(list(y)); ax.set_yticklabels(s.index)
    ax.set_xscale("log"); ax.set_xlabel("Hazard ratio (log scale)")
    for i, (p, hr) in enumerate(zip(s["p"], s["exp(coef)"])):
        ax.text(s["exp(coef) upper 95%"].max() * 1.3, list(y)[i], f"HR={hr:.2f}, p={p:.3f}", va="center")
    fig.tight_layout(); fig.savefig(f"{FIG}/{tag}.pdf", dpi=300); plt.close(fig)


def nomogram(dat, years=(1, 3, 5)):
    """简化列线图：按多因素Cox系数换算为 0-100 分，绘制总分与生存概率关系"""
    cph = CoxPHFitter().fit(dat, "OS.time", "OS", show_progress=False)
    params = cph.params_
    scale = 100 / params.abs().max()
    fig, ax = plt.subplots(figsize=(8, 1.2 * (len(params) + 1) + 1))
    ylabels = list(params.index) + ["Total points"] + [f"{y}-year survival" for y in years]
    for i, name in enumerate(params.index):
        rng = np.linspace(dat[name].min(), dat[name].max(), 5)
        pts = (rng - dat[name].min()) * params[name] * scale
        pts = pts - pts.min()
        ax.scatter(pts, [i] * 5, c="k", s=12)
        for x, v in zip(pts, rng):
            ax.text(x, i - 0.28, f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}", ha="center", fontsize=7)
    # 总分轴与生存率（取均值处的线性预测值换算生存概率）
    total = (dat[params.index] - dat[params.index].min()) * params * scale
    total = total.sum(axis=1)
    tmax = total.max()
    ax.plot([0, tmax], [len(params)] * 2, c="k", lw=.8)
    sf = cph.predict_survival_function(dat.mean().to_frame().T)
    for j, y in enumerate(years):
        t = y * 365
        if t <= sf.index.max():
            s_at = np.interp(t, sf.index, sf.iloc[:, 0].values)
            tot_at = -np.log(max(s_at, 1e-6)) / params.sum() * scale  # 近似换算
            ax.scatter([tot_at], [len(params) + 1 + j], c="k", s=12)
            ax.text(tot_at, len(params) + 1 + j - 0.28, f"{s_at:.2f}", ha="center", fontsize=7)
    ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels)
    ax.set_xlabel("Points"); ax.set_ylim(-0.8, len(ylabels) - 0.2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig4C_nomogram.pdf", dpi=300); plt.close(fig)


def calibration(dat, years=(1, 3, 5)):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    for y, c in zip(years, ["#4DBBD5", "#00A087", "#E64B35"]):
        t = y * 365
        status = ((dat["OS"] == 1) & (dat["OS.time"] <= t)).astype(int)
        valid = dat["OS.time"] > t
        X = pd.DataFrame({"risk": dat["risk"], "age": dat.get("age")}).loc[status == 1]
        X = pd.concat([X, pd.DataFrame({"risk": dat["risk"], "age": dat.get("age")}).loc[valid]])
        yv = np.concatenate([np.ones(X.shape[0] - valid.sum()), np.zeros(valid.sum())])
        if len(np.unique(yv)) < 2: continue
        lr = LogisticRegression().fit(X[["risk"]], yv)
        prob = lr.predict_proba(X[["risk"]])[:, 1]
        bins = pd.qcut(prob, 5, duplicates="drop")
        obs = pd.Series(yv).groupby(bins).mean(); exp = pd.Series(prob).groupby(bins).mean()
        ax.plot(exp, obs, "o-", c=c, label=f"{y}-year")
    ax.plot([0, 1], [0, 1], ls="--", c="grey")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed proportion")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{FIG}/Fig4D_calibration.pdf", dpi=300); plt.close(fig)


def pca_plot(dat):
    """用 4 个模型基因的表达 z 值做 PCA，按风险组着色"""
    from sklearn.decomposition import PCA
    mg = pd.read_csv(f"{OUT}/model_genes.csv")["gene"].tolist()
    expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
    X = expr.loc[mg, dat.index].T
    Z = (X - X.mean()) / X.std(ddof=0)
    pc = PCA(2).fit_transform(Z)
    pc = PCA(2).fit_transform(Z)
    med = dat["risk"].median()
    grp = np.where(dat["risk"] > med, "High risk", "Low risk")
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for g, c in [("High risk", "#E64B35"), ("Low risk", "#4DBBD5")]:
        m = grp == g; ax.scatter(pc[m, 0], pc[m, 1], c=c, s=12, label=g)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig4E_pca.pdf", dpi=300); plt.close(fig)


def main():
    expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
    clin = pd.read_csv(f"{OUT}/tcga_clin.csv").set_index("sample")
    risk = pd.read_csv(f"{OUT}/risk_scores.csv").set_index("sample")
    dat = clin.join(risk).rename(columns={"risk_score": "risk"})
    dat = dat.dropna(subset=["risk", "OS", "OS.time"])
    if "age" in dat.columns:
        dat["age"] = pd.to_numeric(dat["age"], errors="coerce")
        dat = dat[dat["age"] < 120]

    cands = ["risk", "age", "gender", "stage"]
    use = [c for c in cands if dat[c].notna().sum() > 50]
    d = pd.get_dummies(dat[use + ["OS", "OS.time"]], drop_first=True).astype(float)

    uni = CoxPHFitter().fit(d[["risk", "OS", "OS.time"]].dropna(), "OS.time", "OS", show_progress=False)
    forest(uni, "Fig4A_forest_uni")
    multi = CoxPHFitter().fit(d.dropna(), "OS.time", "OS", show_progress=False)
    forest(multi, "Fig4B_forest_multi")
    print("多因素 Cox：\n", multi.summary[["exp(coef)", "p"]])

    nomogram(d.dropna())
    calibration(d.dropna())
    pca_plot(dat.dropna(subset=["risk"]))
    print("04 完成")


if __name__ == "__main__":
    main()
