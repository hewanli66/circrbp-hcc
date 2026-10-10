# -*- coding: utf-8 -*-
"""
11d ESTIMATE 肿瘤纯度分析（二审 Minor 1）
用 Yoshihara 2013 (Nat Commun 4:2612) 补充表中的 Stromal141_UP / Immune141_UP
特征基因集，经 ssGSEA 计算每个 TCGA-LIHC 肿瘤的基质/免疫评分；
基质+免疫评分作为纯度代理（越高=纯度越低）。检验：
  1) 风险评分与 ESTIMATE 评分相关
  2) 关键免疫细胞的 高/低风险组差异在控制 ESTIMATE 后是否仍成立（偏Spearman + 线性模型）
  3) 检查点基因相关在控制 ESTIMATE 后是否仍成立
输出：results/rev11_estimate_purity.csv 等
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, RAW
from scipy import stats
import statsmodels.api as sm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ssgea import ssgsea_from_gmt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module
m11c = import_module("scripts.11c_drug_prolif_confound") if False else None


def partial_spearman(x, y, cov):
    rx, ry, rc = stats.rankdata(x), stats.rankdata(y), stats.rankdata(cov)
    A = np.column_stack([np.ones(len(rc)), rc])
    r_x = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    r_y = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    pr, _ = stats.pearsonr(r_x, r_y)
    n = len(x)
    t = pr * np.sqrt((n - 3) / max(1e-12, 1 - pr ** 2))
    return pr, 2 * stats.t.sf(abs(t), n - 3)


def load_estimate_genes():
    df = pd.read_excel(os.path.join(RAW, "ESTIMATE_moesm488.xlsx"), sheet_name="gene", header=None, skiprows=2)
    df = df[[0, 1]].dropna()
    df.columns = ["gene", "set"]
    out = {}
    for s, sub in df.groupby("set"):
        out[s] = [g for g in sub["gene"] if isinstance(g, str)]
    return out


def main():
    sig = load_estimate_genes()
    print({k: len(v) for k, v in sig.items()})

    expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
    clin = pd.read_csv(f"{OUT}/tcga_clin.csv").set_index("sample")
    risk = pd.read_csv(f"{OUT}/risk_scores.csv").set_index("sample")["risk_score"]

    # 仅原发肿瘤
    tum = [c for c in expr.columns if c[13:15] == "01"]
    E = expr[tum].astype(float)
    scores = ssgsea_from_gmt(E, sig)          # 2 x n
    est = pd.DataFrame({"stromal": scores.loc["Stromal141_UP"],
                        "immune": scores.loc["Immune141_UP"]})
    est["estimate_sum"] = est.stromal + est.immune
    est["risk"] = risk.loc[est.index]
    d = est.join(clin[["OS", "OS.time"]], how="left").dropna(subset=["risk"])
    print(f"n tumors = {len(d)}")

    r1, p1 = stats.spearmanr(d.risk, d.estimate_sum)
    r2, p2 = stats.spearmanr(d.risk, d.stromal)
    r3, p3 = stats.spearmanr(d.risk, d.immune)
    print(f"Spearman(risk, ESTIMATE stromal+immune) = {r1:.3f} p={p1:.1e}")
    print(f"Spearman(risk, stromal) = {r2:.3f}; (risk, immune) = {r3:.3f}")

    # ---- 免疫浸润控制纯度 ----
    immune = pd.read_csv(f"{OUT}/TableS4_immune_infiltration_stats.csv") if os.path.exists(f"{OUT}/TableS4_immune_infiltration_stats.csv") else None
    # 直接读 ssGSEA 免疫细胞评分矩阵
    if os.path.exists(f"{OUT}/immune_ssgsea.csv"):
        infil = pd.read_csv(f"{OUT}/immune_ssgsea.csv", index_col=0)
        print("infiltration matrix:", infil.shape)
    else:
        infil = None
        print("未找到 immune_ssgsea.csv")

    rows = []
    KEY = ["NK cells", "gamma delta T cells", "Cytotoxic cells", "Macrophages",
           "Treg", "Th2 cells", "aDC"]
    if infil is not None:
        for ct in KEY:
            if ct not in infil.index:
                continue
            v = infil.loc[ct, d.index].astype(float)
            ok = v.notna() & d.risk.notna() & d.estimate_sum.notna()
            rho0, p0 = stats.spearmanr(d.risk[ok], v[ok])
            rp, pp = partial_spearman(d.risk[ok].values, v[ok].values, d.estimate_sum[ok].values)
            # 线性模型：浸润 ~ 风险组(中位二分) + ESTIMATE
            grp = (d.risk > d.risk.median()).astype(int)
            X = sm.add_constant(pd.DataFrame({"grp": grp[ok], "est": d.estimate_sum[ok]}))
            lm = sm.OLS(v[ok], X).fit()
            rows.append(dict(cell=ct, rho_raw=rho0, p_raw=p0,
                             rho_partial=rp, p_partial=pp,
                             lm_grp_beta=lm.params["grp"], lm_grp_p=lm.pvalues["grp"],
                             lm_est_beta=lm.params["est"], lm_est_p=lm.pvalues["est"]))
        adj = pd.DataFrame(rows)
        adj.to_csv(f"{OUT}/rev11_purity_immune_adjust.csv", index=False)
        print(adj.to_string(index=False))

    # ---- 检查点控制纯度 ----
    checkpoints = ["PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "CD80", "CD86", "IDO1"]
    cp_rows = []
    for g in checkpoints:
        if g not in expr.index:
            continue
        v = expr.loc[g, d.index].astype(float)
        ok = v.notna() & d.risk.notna() & d.estimate_sum.notna()
        rho0, p0 = stats.spearmanr(d.risk[ok], v[ok])
        rp, pp = partial_spearman(d.risk[ok].values, v[ok].values, d.estimate_sum[ok].values)
        cp_rows.append(dict(gene=g, rho_raw=rho0, p_raw=p0, rho_partial=rp, p_partial=pp))
    cp = pd.DataFrame(cp_rows)
    cp.to_csv(f"{OUT}/rev11_purity_checkpoint_adjust.csv", index=False)
    print(cp.to_string(index=False))

    d.to_csv(f"{OUT}/rev11_estimate_scores.csv")
    print("\n11d 完成")


if __name__ == "__main__":
    main()
