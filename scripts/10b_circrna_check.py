# -*- coding: utf-8 -*-
"""
10b circRNA 公共数据集交叉验证（审稿意见 #1 的证据补充）
数据：GSE94508（Arraystar circRNA 芯片, 5 对 HCC/癌旁）
      GSE97332（Arraystar circRNA 芯片, 7 对 HCC/癌旁）
      （GSE122482 RNA-seq 补充文件到位后可在 FILES 中追加）
分析：
  1) 全局 circRNA 表达负担（每样本全部探针 mean/median log2 强度），配对检验
  2) 探针层面配对差异表达（tumor vs 配对癌旁），统计上调/下调探针数
输出：results/rev1_circrna_summary.csv, results/rev1_circrna_de_<gse>.csv,
      figs/FigS10_circrna_burden.pdf
"""
import sys, os, gzip
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, RAW, COL_RISK, COL_SAFE, FIG_DPI
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

FILES = ["GSE94508_series_matrix.txt.gz", "GSE97332_series_matrix.txt.gz"]


def load_series_matrix(path):
    """解析 GEO series matrix -> (表达DataFrame[探针x样本], 样本分组Series)"""
    meta, rows = {}, []
    with gzip.open(path, "rt", errors="ignore") as f:
        in_table = False
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                meta["gsm"] = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                meta["title"] = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_source_name_ch1"):
                meta["source"] = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                break
            elif in_table:
                parts = line.rstrip("\n").split("\t")
                rows.append(parts)
    header = [h.strip('"') for h in rows[0]]
    df = pd.DataFrame(rows[1:], columns=header).set_index("ID_REF")
    df = df.apply(pd.to_numeric, errors="coerce")
    grp = pd.Series(meta["source"], index=meta["gsm"], name="group")
    return df, grp, meta["title"]


def main():
    summary_rows, fig_data = [], {}
    for fname in FILES:
        gse = fname.split("_")[0]
        df, grp, titles = load_series_matrix(os.path.join(RAW, fname))
        df = df.dropna(how="all")
        print(f"\n{gse}: {df.shape[0]} 探针 x {df.shape[1]} 样本")
        print(grp.value_counts().to_string())

        X = np.log2(df + 1.0)                       # log2 强度
        tsel = grp[grp.str.contains("carcinoma|cancer|tumor", case=False, na=False)
                   & ~grp.str.contains("para", case=False, na=False)]
        nsel = grp[grp.str.contains("para|non-tumor|normal|adjacent", case=False, na=False)]
        # 按标题排序后位置配对（兼容 "1-1/1-2" 与 "...circRNA 1/2" 两种命名）
        tm = dict(zip(grp.index, titles))
        tumor_cols = sorted(tsel.index, key=lambda c: tm[c])
        normal_cols = sorted(nsel.index, key=lambda c: tm[c])
        pairs = list(zip(tumor_cols, normal_cols))
        print(f"配对样本: {len(pairs)} 对")
        print("  tumor:", [tm[c] for c in tumor_cols])
        print("  normal:", [tm[c] for c in normal_cols])

        burden_t = X[[c for c, _ in pairs]].mean(axis=0).values
        burden_n = X[[c for _, c in pairs]].mean(axis=0).values
        t_burden, p_burden = stats.ttest_rel(burden_t, burden_n)
        fig_data[gse] = (burden_t, burden_n)

        # 探针层面配对 DE
        de_rows = []
        for probe in X.index:
            vt = X.loc[probe, [c for c, _ in pairs]].values.astype(float)
            vn = X.loc[probe, [c for _, c in pairs]].values.astype(float)
            if np.isnan(vt).any() or np.isnan(vn).any():
                continue
            tt, pp = stats.ttest_rel(vt, vn)
            de_rows.append((probe, float(vt.mean() - vn.mean()), tt, pp))
        de = pd.DataFrame(de_rows, columns=["probe", "log2FC", "t", "p"])
        de["fdr"] = np.minimum.accumulate((de.p * len(de) / (de.p.rank()))[de.p.sort_values().index])[::-1]
        de = de.sort_values("p")
        de.to_csv(f"{OUT}/rev1_circrna_de_{gse}.csv", index=False)
        n_up = int(((de.p < 0.05) & (de.log2FC > 0)).sum())
        n_dn = int(((de.p < 0.05) & (de.log2FC < 0)).sum())
        sig = de[de.fdr < 0.05]
        summary_rows.append((gse, df.shape[0], len(pairs), float(burden_t.mean()), float(burden_n.mean()),
                             float(t_burden), float(p_burden), n_up, n_dn,
                             int((sig.log2FC > 0).sum()), int((sig.log2FC < 0).sum())))

    summ = pd.DataFrame(summary_rows, columns=["dataset", "n_probes", "n_pairs", "burden_tumor",
                                               "burden_normal", "burden_t", "burden_p",
                                               "n_up_p0.05", "n_dn_p0.05", "n_up_fdr0.05", "n_dn_fdr0.05"])
    summ.to_csv(f"{OUT}/rev1_circrna_summary.csv", index=False)
    print("\n汇总:")
    print(summ.to_string(index=False))

    fig, axes = plt.subplots(1, len(fig_data), figsize=(4.6 * len(fig_data), 4.2), squeeze=False)
    for ax, (gse, (bt, bn)) in zip(axes[0], fig_data.items()):
        bp = ax.boxplot([bn, bt], labels=["Non-tumor", "HCC"], patch_artist=True, widths=0.5,
                        medianprops=dict(color="black"))
        for p, c in zip(bp["boxes"], [COL_SAFE, COL_RISK]):
            p.set_facecolor(c); p.set_alpha(0.85)
        for i in range(len(bt)):   # 配对连线
            ax.plot([1, 2], [bn[i], bt[i]], color="grey", lw=0.7, alpha=0.6)
        _, p = stats.ttest_rel(bt, bn)
        ax.set_title(f"{gse}\npaired t-test p={p:.3e}", fontsize=10)
        ax.set_ylabel("Mean log2 circRNA intensity (all probes)")
    fig.suptitle("Global circRNA abundance in HCC vs matched non-tumor liver", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/FigS10_circrna_burden.pdf", dpi=FIG_DPI); plt.close(fig)
    print("\n10b 完成")


if __name__ == "__main__":
    main()
