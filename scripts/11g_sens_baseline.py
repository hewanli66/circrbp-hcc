# -*- coding: utf-8 -*-
"""
11g 二审 Minor 2 / Minor 10
Minor 10: 剔除 2 例复发肿瘤（sample type 02）后的 TCGA 敏感性分析
Minor 2 : TCGA vs ICGC LIRI-JP 基线特征对比表（年龄/性别/分期/病因）
输出：results/rev13_excl_recurrent.csv, results/rev13_baseline_table.csv
"""
import sys, os
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from sksurv.metrics import concordance_index_censored

# ---------- Minor 10: 剔除复发肿瘤的敏感性分析 ----------
clin = pd.read_csv(f"{OUT}/tcga_clin.csv")
risk = pd.read_csv(f"{OUT}/risk_scores.csv").set_index("sample")["risk_score"]

def eval_set(d):
    d = d.dropna(subset=["OS", "OS.time", "risk"])
    d = d[d["OS.time"] >= 30]
    med = d.risk.median()
    hi, lo = d[d.risk > med], d[d.risk <= med]
    p = logrank_test(hi["OS.time"], lo["OS.time"], hi.OS, lo.OS).p_value
    ci = concordance_index_censored(d.OS.astype(bool).values, d["OS.time"].values, d.risk.values)[0]
    cph = CoxPHFitter().fit(d[["risk", "OS.time", "OS"]], "OS.time", "OS")
    return dict(n=len(d), events=int(d.OS.sum()), logrank_p=p, c_index=ci,
                HR=float(cph.hazard_ratios_["risk"]), HR_p=float(cph.summary.p.iloc[0]))

d_all = clin.set_index("sample").join(risk.rename("risk"), how="inner").reset_index()
tum_only = d_all[d_all["sample"].str[13:15].isin(["01"])]   # 仅343原发
res_all = eval_set(d_all); res_343 = eval_set(tum_only)
print("全393:", res_all)
print("仅343原发(剔除2例复发):", res_343)
pd.DataFrame([dict(version="all_393", **res_all), dict(version="primary_only_343", **res_343)]).to_csv(
    f"{OUT}/rev13_excl_recurrent.csv", index=False)

# ---------- Minor 2: 基线对比 ----------
# TCGA
tc = d_all[d_all["sample"].str[13:15].isin(["01", "02"])]
base_rows = []
def add(name, age, gender, stage_ser=None, virus_ser=None):
    base_rows.append(dict(cohort=name, n=len(age),
                          age_median=float(np.nanmedian(age)), age_iqr=f"{np.nanpercentile(age,25):.0f}-{np.nanpercentile(age,75):.0f}",
                          male_pct=round((gender.astype(str).str.lower() == "male").mean() * 100, 1)))
add("TCGA-LIHC", tc["age"].values, tc["gender"])
stg = tc["stage"].astype(str).str.extract(r"([IV]+)")[0].value_counts(normalize=True)
print("TCGA stage dist:", (stg * 100).round(1).to_dict())

# LIRI-JP
pat = pd.read_csv(os.path.join(os.path.dirname(OUT), "rawdata", "hccdb18_patient", "HCCDB18.patient.txt"),
                  sep="\t", index_col=0)
def row(idx):
    return pat.loc[idx].astype(str)
# 仅保留进入分析的 212 例（按 10a 的输出约束无法完全回溯，用全队列并注明）
age = pd.to_numeric(row("AGE"), errors="coerce")
gender = row("GENDER")
liri = pd.DataFrame({"age": age, "gender": gender})
liri = liri.dropna(subset=["age"])
add("ICGC LIRI-JP (all with age)", liri["age"].values, liri["gender"])
vir = row("VIRUS").value_counts(normalize=True)
stg2 = row("TNM_STAGE_T1").value_counts(normalize=True)
print("LIRI virus:", (vir * 100).round(1).to_dict())
print("LIRI stage:", (stg2 * 100).round(1).to_dict())
base = pd.DataFrame(base_rows)
base.to_csv(f"{OUT}/rev13_baseline_table.csv", index=False)
print(base.to_string(index=False))
print("\n11g 完成")
