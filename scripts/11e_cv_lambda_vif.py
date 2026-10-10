# -*- coding: utf-8 -*-
"""
11e 二审 Minor 5/6 分析
Minor 5: LASSO 正则化路径的交叉验证曲线 + CV最优λ/1-SEλ对应基因数；
         以03脚本同款流程重算 λ=0.0408 处 5 折 CV C-index（统一 0.605/0.619 口径）
Minor 6: 风险分与 G2M 元基因的共线性诊断(VIF)；
         TCGA 中 签名 vs 元基因 C-index 差异的 bootstrap CI
输出：results/rev12_cv_lambda.csv, rev12_cv_summary.txt, rev12_vif_cindex.csv
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import KFold
from scipy import stats
import statsmodels.api as sm

LAM_SEL = 0.0408


def load_tcga():
    expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
    clin = pd.read_csv(f"{OUT}/tcga_clin.csv").set_index("sample")
    cand = pd.read_csv(f"{OUT}/univ_cox.csv").gene.tolist()
    sub = expr.loc[cand, clin.index].T
    sub_z = (sub - sub.mean()) / sub.std(ddof=0)
    y = Surv.from_arrays(clin["OS"].astype(bool).values, clin["OS.time"].values)
    return sub_z, y, clin


def main():
    sub_z, y, clin = load_tcga()
    genes = sub_z.columns.tolist()
    kf = KFold(5, shuffle=True, random_state=42)

    # ---------- CV 曲线（全路径） ----------
    cox = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=60)
    cox.fit(sub_z.values, y)
    rows = []
    for li, lam in enumerate(cox.alphas_):
        n_genes = int((cox.coef_[:, li] != 0).sum())
        scs = []
        for tr, va in kf.split(sub_z.values):
            try:
                m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[lam]).fit(sub_z.values[tr], y[tr])
                scs.append(concordance_index_censored(y[va]["event"], y[va]["time"],
                                                      m.predict(sub_z.values[va]))[0])
            except Exception:
                pass
        rows.append(dict(lambda_=lam, n_genes=n_genes, cv_cindex=np.mean(scs) if scs else np.nan,
                         cv_se=stats.sem(scs) if len(scs) > 1 else np.nan))
    cv = pd.DataFrame(rows)
    cv.to_csv(f"{OUT}/rev12_cv_lambda.csv", index=False)

    best = cv.loc[cv.cv_cindex.idxmax()]
    thr = best.cv_cindex - best.cv_se
    one_se = cv[cv.cv_cindex >= thr].iloc[0]   # 最大λ且CV在1-SE内
    sel_row = cv[np.isclose(cv.lambda_, LAM_SEL, atol=1e-6)]
    sel = sel_row.iloc[0] if len(sel_row) else cv.iloc[(cv.lambda_ - LAM_SEL).abs().idxmin()]
    lines = [
        f"n_alphas={len(cv)}, lambda range {cv.lambda_.max():.4f} ~ {cv.lambda_.min():.4f}",
        f"CV-optimal: lambda={best.lambda_:.4f}, n_genes={int(best.n_genes)}, cv_cindex={best.cv_cindex:.3f}",
        f"1-SE rule:  lambda={one_se.lambda_:.4f}, n_genes={int(one_se.n_genes)}, cv_cindex={one_se.cv_cindex:.3f}",
        f"Selected:   lambda={sel.lambda_:.4f}, n_genes={int(sel.n_genes)}, cv_cindex={sel.cv_cindex:.3f}",
        f"full-model apparent C-index at selected lambda (TCGA, all 393): see 09a/rev3",
    ]
    open(f"{OUT}/rev12_cv_summary.txt", "w").write("\n".join(lines))
    print("\n".join(lines))

    # ---------- Minor 6: VIF + C-index 差 bootstrap ----------
    g2m = None
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "genesets", "h.all.v7.5.1.symbols.gmt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("HALLMARK_G2M_CHECKPOINT"):
                g2m = set(line.rstrip("\n").split("\t")[2:])
    expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
    risk = pd.read_csv(f"{OUT}/risk_scores.csv").set_index("sample")["risk_score"]
    g2 = [g for g in g2m if g in expr.index]
    sub = expr.loc[g2, clin.index].astype(float)
    z = (sub.T - sub.T.mean()) / sub.T.std(ddof=0)
    prolif = z.mean(axis=1)
    d = pd.DataFrame({"risk": risk.loc[clin.index], "prolif": prolif.loc[clin.index],
                      "OS": clin["OS"].values, "OS.time": clin["OS.time"].values}).dropna()

    X = sm.add_constant(d[["risk", "prolif"]])
    vifs = {}
    for i, col in enumerate(["risk", "prolif"]):
        others = [c for c in ["risk", "prolif"] if c != col]
        r2 = sm.OLS(d[col], sm.add_constant(d[others])).fit().rsquared
        vifs[col] = 1 / (1 - r2)
    rho_rp = stats.spearmanr(d.risk, d.prolif)[0]
    print(f"Spearman(risk, prolif)={rho_rp:.3f}; Pearson R2={sm.OLS(d.risk, sm.add_constant(d[['prolif']])).fit().rsquared:.3f}; VIF={vifs}")

    ev_b = d["OS"].astype(bool).values
    t_b = d["OS.time"].values
    ci_r = concordance_index_censored(ev_b, t_b, d.risk.values)[0]
    ci_m = concordance_index_censored(ev_b, t_b, d.prolif.values)[0]
    rng = np.random.default_rng(42)
    diffs = []
    n = len(d)
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        if ev_b[idx].sum() < 10:
            continue
        a = concordance_index_censored(ev_b[idx], t_b[idx], d.risk.values[idx])[0]
        b = concordance_index_censored(ev_b[idx], t_b[idx], d.prolif.values[idx])[0]
        diffs.append(a - b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out = pd.DataFrame([{"ci_signature": ci_r, "ci_metagene": ci_m, "diff": ci_r - ci_m,
                         "diff_ci95_lo": lo, "diff_ci95_hi": hi,
                         "vif_risk": vifs["risk"], "vif_prolif": vifs["prolif"],
                         "spearman_risk_prolif": rho_rp}])
    out.to_csv(f"{OUT}/rev12_vif_cindex.csv", index=False)
    print(out.to_string(index=False))
    print("\n11e 完成")


if __name__ == "__main__":
    main()
