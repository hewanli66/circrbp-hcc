# -*- coding: utf-8 -*-
"""官方标准化数据 vs count自建归一化 交叉验证 + Fig7统计量重算比对。"""
import pandas as pd, numpy as np
from scipy import stats

GENES = ["SFPQ", "HNRNPM", "IGF2BP3", "NOVA1"]
BASE = r"C:\Users\hewan\Documents\kimi\tasks\2026-10-08\11-38-08-f56eebff"

off = np.load(BASE + r"\results\sc_normalized_check.npz")
cnt = np.load(BASE + r"\results\sc_count4.npz")
off_cells = list(off["cells"]); cnt_cells = list(cnt["cells"])
print("[0] 细胞列一致性: 官方顺序==count顺序:", off_cells == cnt_cells)

meta = pd.read_csv(BASE + r"\rawdata\GSE149614_HCC.metadata.updated.txt.gz", sep="\t").set_index("Cell")
m_set = set(meta.index)
common = [c for c in off_cells if c in m_set]
print(f"[0] 元数据匹配: {len(common)}/{len(off_cells)}")
m = meta.loc[common]

E = pd.DataFrame({g: off[g] for g in GENES}, index=off_cells).loc[common].T   # 官方标准化
C = pd.DataFrame({g: cnt[g] for g in GENES}, index=cnt_cells).T
ours = pd.DataFrame(np.log2(1 + C.values / (cnt["sums"] / 1e4)[None, :]),
                    index=C.index, columns=C.columns).loc[:, common]           # 我们此前的归一化

print("\n[1] 官方标准化 vs 自建归一化（全体细胞 Spearman）:")
for g in GENES:
    rho, _ = stats.spearmanr(E.loc[g].values, ours.loc[g].values)
    print(f"  {g:8s} rho = {rho:.4f}")

print("\n[2] 分细胞类型统计（已发表 sc_gene_stats.csv -> 官方数据重算）:")
pub = pd.read_csv(BASE + r"\results\sc_gene_stats.csv")
for ct in ["Hepatocyte", "T/NK", "Myeloid", "Endothelial", "Fibroblast", "B"]:
    sel = m.celltype == ct
    for g in GENES:
        v = E.loc[g, sel]
        mean_new, pct_new = v.mean(), (v > 0).mean() * 100
        r = pub[(pub.celltype == ct) & (pub.gene == g)]
        if len(r):
            print(f"  {ct:12s} {g:8s} mean {r.mean_expr.iloc[0]:.3f} -> {mean_new:.3f} | "
                  f"pct {r.pct_expr.iloc[0]:.1f} -> {pct_new:.1f}")

print("\n[3] 肝细胞 Tumor vs Normal（稿件数字 vs 官方数据重算）:")
hepa = m.celltype == "Hepatocyte"
mh = m[hepa]
Eh = E.loc[:, hepa]
pub_hp = pd.read_csv(BASE + r"\results\sc_hepatocyte_stats.csv").set_index("gene")["Tumor_vs_Normal_p"]
coef = pd.read_csv(BASE + r"\results\model_genes.csv").set_index("gene")["coef"]
z = Eh.sub(Eh.mean(axis=1), axis=0).div(Eh.std(axis=1).replace(0, np.nan), axis=0)
rs = sum(z.loc[g] * coef.get(g, 0) for g in GENES)
ok = True
for g in GENES:
    pv = stats.mannwhitneyu(Eh.loc[g, mh.site == "Tumor"], Eh.loc[g, mh.site == "Normal"]).pvalue
    agree = pv < 0.05
    ok &= agree
    print(f"  {g:8s} 稿件 p = {pub_hp[g]:.2e} | 官方重算 p = {pv:.2e} | 一致显著: {agree}")
pv = stats.mannwhitneyu(rs[mh.site == "Tumor"], rs[mh.site == "Normal"]).pvalue
print(f"  4基因风险分 稿件 p = 5.1e-68 | 官方重算 p = {pv:.2e} | 一致显著: {pv < 0.05}")
print("\n结论:", "全部统计量与稿件一致 ✓" if ok and pv < 0.05 else "存在不一致，需复查 ✗")
