# -*- coding: utf-8 -*-
"""
10a ICGC LIRI-JP 独立外验证（HCCDB18 通道）
数据：rawdata/HCCDB18_mRNA_level3.zip  (log2 标准化表达, 212 HCC + 177 癌旁)
      rawdata/HCCDB18.sample.zip        (样本-患者映射, TYPE=HCC/Adjacent)
      rawdata/HCCDB18.patient.zip       (STATUS=Alive/Dead, SUR=生存月数)
方法：TCGA 拟合的 4 基因 LASSO 系数直接套用（LIRI-JP 肿瘤样本内部 z 分数），
      按中位数二分，log-rank + C-index + 连续 HR。
输出：results/icgc_lirijp_validation.csv, figs/FigS9_km_icgc.pdf
"""
import sys, os, zipfile, io
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, RAW, COL_RISK, COL_SAFE, FIG_DPI
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"


def read_zip_txt(name):
    with zipfile.ZipFile(os.path.join(RAW, name)) as z:
        f = z.namelist()[0]
        return pd.read_csv(z.open(f), sep="\t")


def main():
    # ---------- 表达矩阵 ----------
    expr = read_zip_txt("HCCDB18_mRNA_level3.zip")
    expr = expr[~expr.Symbol.isna()].set_index("Symbol")
    expr = expr.drop(columns=["Entrez_ID"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    expr = expr[~expr.index.duplicated(keep="first")]
    print(f"表达矩阵: {expr.shape[0]} 基因 x {expr.shape[1]} 样本")

    # ---------- 样本信息（转置键值表）----------
    sinfo = read_zip_txt("HCCDB18.sample.zip").set_index("SAMPLE_ID").T
    sinfo.index.name = "sample"
    # ---------- 患者信息 ----------
    pinfo = read_zip_txt("HCCDB18.patient.zip").set_index("PATIENT_ID").T
    pinfo.index.name = "patient"
    surv = pinfo[["STATUS", "SUR"]].copy()
    surv["OS"] = (surv.STATUS == "Dead").astype(int)
    surv["OS.time"] = pd.to_numeric(surv.SUR, errors="coerce") * 30.44
    print(f"患者: {len(surv)}, 死亡 {surv.OS.sum()}, 生存月数范围 "
          f"{pd.to_numeric(surv.SUR, errors='coerce').min():.0f}-"
          f"{pd.to_numeric(surv.SUR, errors='coerce').max():.0f}")

    # 样本 -> 患者 -> 生存
    dat = sinfo.join(surv, on="PATIENT_ID")
    hcc = dat[dat.TYPE == "HCC"].copy()
    print(f"HCC 样本: {len(hcc)}（癌旁 {int((dat.TYPE == 'Adjacent').sum())} 不进入生存验证）")

    # ---------- 风险评分 ----------
    coef = pd.read_csv(f"{OUT}/model_genes.csv").set_index("gene")["coef"]
    g4 = [g for g in coef.index if g in expr.index]
    assert len(g4) == 4, f"基因缺失: {set(coef.index) - set(g4)}"
    sub = expr.loc[g4, hcc.index].T.astype(float)
    sub = sub.loc[:, sub.std(ddof=0) > 0]
    z = (sub - sub.mean()) / sub.std(ddof=0)
    hcc["risk"] = (z * coef[sub.columns]).sum(axis=1)

    d = hcc.dropna(subset=["OS", "OS.time", "risk"])
    d = d[d["OS.time"] >= 30]
    print(f"进入分析: {len(d)} 例, 事件 {int(d.OS.sum())}")

    ci = concordance_index_censored(d.OS.astype(bool).values, d["OS.time"].values, d.risk.values)[0]
    cph = CoxPHFitter().fit(d[["risk", "OS.time", "OS"]], "OS.time", "OS")
    hr = np.exp(cph.params_["risk"]); hrp = cph.summary.loc["risk", "p"]
    med = d.risk.median()
    hi, lo = d[d.risk > med], d[d.risk <= med]
    p = logrank_test(hi["OS.time"], lo["OS.time"], hi.OS, lo.OS).p_value

    res = pd.DataFrame([{"cohort": "ICGC_LIRI-JP_OS", "n": len(d), "events": int(d.OS.sum()),
                         "c_index": ci, "HR_cont": hr, "HR_p": hrp, "logrank_p": p,
                         "cutoff": "median"}])
    res.to_csv(f"{OUT}/icgc_lirijp_validation.csv", index=False)
    print(res.to_string(index=False))

    # KM 图
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    kf = KaplanMeierFitter()
    for grp, dd, c, lb in [("High risk", hi, COL_RISK, f"High (n={len(hi)})"),
                           ("Low risk", lo, COL_SAFE, f"Low (n={len(lo)})")]:
        kf.fit(dd["OS.time"], dd.OS, label=lb)
        kf.plot_survival_function(ax=ax, ci_show=False, c=c)
    ax.set_title(f"ICGC LIRI-JP validation\nlog-rank p={p:.3e}", fontsize=11)
    ax.set_xlabel("Days"); ax.set_ylabel("Survival probability")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{FIG}/FigS9_km_icgc.pdf", dpi=FIG_DPI); plt.close(fig)
    print("10a 完成")


if __name__ == "__main__":
    main()
