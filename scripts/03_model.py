# -*- coding: utf-8 -*-
"""
03 预后相关性分析（备选方案框架下的一个组成部分）
流程：Coxnet LASSO 全路径 -> 选取 4 基因解（简约性与交叉验证判别力的平衡）
      -> 风险评分 -> TCGA 评估 + GSE14520（OS 与 RFS）+ GSE76427 验证
输出：
  figs/Fig2A_igf2bp3_km.pdf     IGF2BP3 高低表达 KM（TCGA）
  figs/Fig2B_km_tcga.pdf        4基因风险评分 KM（TCGA）
  figs/Fig2C_roc_tcga.pdf       1/3/5年 ROC（TCGA）
  figs/Fig2D_risk_triplot.pdf   三联图
  figs/Fig2E_km_gse14520_os.pdf / Fig2F_km_gse14520_rfs.pdf
  figs/Fig2G_km_gse76427.pdf
  results/model_genes.csv, results/risk_scores.csv, results/model_validation.csv
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, RAW, COL_GROUP, COL_RISK, COL_SAFE
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_curve, auc
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

N_GENES = 4  # 从 LASSO 路径中选取 4 基因解


def km_ax(ax, d, score, title, time_col="OS.time", event_col="OS"):
    med = score.median()
    hi, lo = d[score > med], d[score <= med]
    kf = KaplanMeierFitter()
    for grp, df_, c in [("High", hi, COL_RISK), ("Low", lo, COL_SAFE)]:
        kf.fit(df_[time_col], df_[event_col], label=grp)
        kf.plot_survival_function(ax=ax, ci_show=False, c=c)
    p = logrank_test(hi[time_col], lo[time_col], hi[event_col], lo[event_col]).p_value
    ax.set_title(f"{title}\nlog-rank p={p:.3e}", fontsize=10)
    ax.set_xlabel("Days"); ax.set_ylabel("Survival probability")
    return p


def roc_ax(ax, d, score, title, time_col="OS.time", event_col="OS"):
    for yr, c in [(1, COL_GROUP[0]), (3, "#00A087"), (5, COL_GROUP[1])]:
        status = ((d[event_col] == 1) & (d[time_col] <= yr * 365)).astype(int)
        if status.nunique() < 2: continue
        fpr, tpr, _ = roc_curve(status, score)
        ax.plot(fpr, tpr, c=c, label=f"{yr}-yr AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", c="grey")
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(title, fontsize=10); ax.legend(frameon=False, fontsize=8)


def geo_risk(ge, gc, coef, event_col, time_col):
    """GEO 队列风险评分与临床表合并"""
    ge = ge[~ge.index.duplicated()]
    common = [g for g in coef.index if g in ge.index]
    gd = ge.loc[common].T
    gd = gd.loc[:, gd.std(ddof=0) > 0]
    z = (gd - gd.mean()) / gd.std(ddof=0)
    grs = (z * coef[gd.columns]).sum(axis=1)
    gdat = pd.concat([grs.rename("risk"), gc.set_index("sample")], axis=1)
    gdat = gdat.dropna(subset=[event_col, time_col, "risk"])
    gdat = gdat[gdat[time_col] >= 30]
    return gdat, len(common)


def main():
    expr = pd.read_csv(f"{OUT}/tcga_expr.csv", index_col=0)
    clin = pd.read_csv(f"{OUT}/tcga_clin.csv").set_index("sample")
    cand = pd.read_csv(f"{OUT}/univ_cox.csv").gene.tolist()
    sub = expr.loc[cand, clin.index].T
    sub_z = (sub - sub.mean()) / sub.std(ddof=0)
    y = Surv.from_arrays(clin["OS"].astype(bool).values, clin["OS.time"].values)

    cox = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.01, n_alphas=60)
    cox.fit(sub_z.values, y)
    # 从路径中选 4 基因解
    coef = None
    for li, lam in enumerate(cox.alphas_):
        genes = [g for g, b in zip(sub_z.columns, cox.coef_[:, li] != 0) if b]
        if len(genes) == N_GENES:
            coef = pd.Series({g: float(cox.coef_[list(sub_z.columns).index(g), li]) for g in genes})
            print(f"选定 lambda={lam:.4f} 处的 {N_GENES} 基因解: {list(coef.index)}")
            break
    if coef is None:
        raise RuntimeError("路径中未找到 4 基因解，请调整 N_GENES 或 alpha 网格")

    # 交叉验证 C-index
    from sklearn.model_selection import KFold
    kf = KFold(5, shuffle=True, random_state=42)
    sc = []
    for tr, va in kf.split(sub_z.values):
        m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[lam]).fit(sub_z.values[tr], y[tr])
        sc.append(concordance_index_censored(y[va]["event"], y[va]["time"], m.predict(sub_z.values[va]))[0])
    print(f"5折CV C-index = {np.mean(sc):.3f}")
    pd.DataFrame({"gene": coef.index, "coef": coef.values}).to_csv(f"{OUT}/model_genes.csv", index=False)

    rs = (sub_z * coef).sum(axis=1)
    d = clin.copy()
    d["risk"] = rs

    # Fig2A: 4 个模型基因在肿瘤 vs 正常中的表达（IGF2BP3 为核心亮点）
    from scipy import stats as _st
    genes4 = list(coef.index)
    tum = clin.index.str.endswith("-01")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, gn in zip(axes, genes4):
        v_t = expr.loc[gn, clin.index[tum]]
        v_n = expr.loc[gn, clin.index[clin.index.str.endswith("-11")]]
        bp = ax.boxplot([v_n, v_t], labels=["Normal", "Tumor"], patch_artist=True, widths=0.55,
                        medianprops=dict(color="black"))
        for p, c in zip(bp["boxes"], [COL_SAFE, COL_RISK]):
            p.set_facecolor(c); p.set_alpha(0.85)
        p = _st.ttest_ind(v_t, v_n, equal_var=False).pvalue
        ax.set_title(f"{gn}\nWelch p={p:.1e}", fontsize=10)
        ax.set_ylabel("log2(FPKM+1)" if gn == genes4[0] else "")
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig2A_model_genes_expr.pdf", dpi=300); plt.close(fig)

    # Fig2B/C: TCGA 风险评分 KM + ROC
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    p_tcga = km_ax(axes[0], d, d["risk"], "circRBP risk score (TCGA)")
    roc_ax(axes[1], d, d["risk"], "ROC (TCGA)")
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig2B_C_tcga.pdf", dpi=300); plt.close(fig)

    # Fig2D: 三联图
    order = np.argsort(d["risk"])
    med = d["risk"].median()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    axes[0].scatter(range(len(d)), d["risk"].iloc[order], c=np.where(d["risk"].iloc[order] > med, COL_RISK, COL_SAFE), s=8)
    axes[0].axhline(med, ls="--", c="k", lw=.8); axes[0].set_xlabel("Patients (ranked)"); axes[0].set_ylabel("Risk score")
    axes[1].scatter(range(len(d)), d["OS.time"].iloc[order], c=np.where(d["OS"].iloc[order] == 1, COL_RISK, COL_SAFE), s=8)
    axes[1].set_xlabel("Patients (ranked)"); axes[1].set_ylabel("Survival (days)")
    from seaborn import heatmap
    heatmap(sub_z.loc[d.index].iloc[order][list(coef.index)].T, cmap="RdBu_r", center=0, ax=axes[2],
            cbar_kws={"label": "z-score"})
    axes[2].set_xlabel("Patients (ranked)")
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig2D_risk_triplot.pdf", dpi=300); plt.close(fig)

    pd.DataFrame({"sample": d.index, "risk_score": rs.values}).to_csv(f"{OUT}/risk_scores.csv", index=False)

    # ---------- GEO 验证 ----------
    val_rows = [("TCGA", len(d), int(d.OS.sum()), p_tcga)]
    ge14 = pd.read_csv(f"{OUT}/gse14520_expr_symbol.csv", index_col=0)
    gc14 = pd.read_csv(f"{OUT}/gse14520_clin_raw.csv")
    g14, n14 = geo_risk(ge14, gc14, coef, "OS", "OS.time")
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    p14 = km_ax(ax, g14, g14["risk"], "circRBP risk score (GSE14520, OS)")
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig2E_km_gse14520_os.pdf", dpi=300); plt.close(fig)
    val_rows.append(("GSE14520_OS", len(g14), int(g14.OS.sum()), p14))

    # GSE14520 RFS（从原始补充表重取复发数据）
    sup = pd.read_csv(os.path.join(RAW, "GSE14520_Extra_Supplement.txt.gz"), sep="\t")
    sup.columns = [c.strip() for c in sup.columns]
    rfs = sup.rename(columns={"Recurr months": "RFS.time", "Recurr status": "RFS"})[["Affy_GSM", "RFS.time", "RFS"]].dropna()
    rfs["sample"] = rfs["Affy_GSM"].astype(str).str.strip()
    rfs["RFS.time"] = pd.to_numeric(rfs["RFS.time"], errors="coerce") * 30.44
    rfs["RFS"] = pd.to_numeric(rfs["RFS"], errors="coerce")
    g14r, _ = geo_risk(ge14, rfs, coef, "RFS", "RFS.time")
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    p14r = km_ax(ax, g14r, g14r["risk"], "circRBP risk score (GSE14520, RFS)", "RFS.time", "RFS")
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig2F_km_gse14520_rfs.pdf", dpi=300); plt.close(fig)
    val_rows.append(("GSE14520_RFS", len(g14r), int(g14r.RFS.sum()), p14r))

    ge76 = pd.read_csv(f"{OUT}/gse76427_expr_symbol.csv", index_col=0)
    gc76 = pd.read_csv(f"{OUT}/gse76427_clin_raw.csv")
    g76, n76 = geo_risk(ge76, gc76, coef, "OS", "OS.time")
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    p76 = km_ax(ax, g76, g76["risk"], "circRBP risk score (GSE76427)")
    fig.tight_layout(); fig.savefig(f"{FIG}/Fig2G_km_gse76427.pdf", dpi=300); plt.close(fig)
    val_rows.append(("GSE76427_OS", len(g76), int(g76.OS.sum()), p76))

    vd = pd.DataFrame(val_rows, columns=["cohort", "n", "events", "logrank_p"])
    vd.to_csv(f"{OUT}/model_validation.csv", index=False)
    print(vd.to_string(index=False))
    print("03 完成")


if __name__ == "__main__":
    main()
