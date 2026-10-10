# -*- coding: utf-8 -*-
"""
11a circRNA 交叉验证 v2（二审 Major 1 整改）
整改点：
  1) 逐数据集分别处理量纲（检测原始线性强度 vs 已log2），不做跨数据集合并比较
  2) 全局丰度改用中位数（稳健）+ 均值双指标；配对差异给 95% CI（bootstrap pairs）
  3) 效能分析：报告当前 n 下对 dz=0.5/0.8 的检验效能 + 达到 80% 效能所需 n
  4) 报告 Cohen's dz 与两个数据集的方向一致性
  5) 明确记录：circRNA 芯片不含 mRNA 通道 -> circRBP-circRNA 相关无法在本数据中计算（写入输出供回复信引用）
输出：results/rev1b_circrna_pilot.csv, results/rev1b_power.txt, figs/FigS10_circrna_burden.pdf(重绘)
"""
import sys, os, gzip
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG, RAW, COL_RISK, COL_SAFE, FIG_DPI
from scipy import stats
from statsmodels.stats.power import TTestPower
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = ["GSE94508_series_matrix.txt.gz", "GSE97332_series_matrix.txt.gz"]


def load_series_matrix(path):
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
                rows.append(line.rstrip("\n").split("\t"))
    header = [h.strip('"') for h in rows[0]]
    df = pd.DataFrame(rows[1:], columns=header).set_index("ID_REF")
    df = df.apply(pd.to_numeric, errors="coerce")
    grp = pd.Series(meta["source"], index=meta["gsm"], name="group")
    return df, grp, meta["title"]


def boot_ci(diff, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
    return np.percentile(means, [2.5, 97.5])


def main():
    rows, fig_data = [], {}
    power_lines = []
    for fname in FILES:
        gse = fname.split("_")[0]
        df, grp, titles = load_series_matrix(os.path.join(RAW, fname))
        df = df.dropna(how="all")

        raw_vals = df.values.astype(float)
        raw_vals = raw_vals[~np.isnan(raw_vals)]
        scale = "linear(raw)" if np.nanmax(raw_vals) > 100 else "log2"
        X = np.log2(df + 1.0) if scale == "linear(raw)" else df.copy()

        tsel = grp[grp.str.contains("carcinoma|cancer|tumor", case=False, na=False)
                   & ~grp.str.contains("para", case=False, na=False)]
        nsel = grp[grp.str.contains("para|non-tumor|normal|adjacent", case=False, na=False)]
        tm = dict(zip(grp.index, titles))
        tumor_cols = sorted(tsel.index, key=lambda c: tm[c])
        normal_cols = sorted(nsel.index, key=lambda c: tm[c])
        pairs = list(zip(tumor_cols, normal_cols))
        n = len(pairs)

        bt = X[[c for c, _ in pairs]].median(axis=0).values   # 每样本中位数
        bn = X[[c for _, c in pairs]].median(axis=0).values
        diff = bt - bn
        dz = diff.mean() / diff.std(ddof=1)
        ci = boot_ci(diff)
        t_p = stats.ttest_rel(bt, bn).pvalue
        w_p = stats.wilcoxon(bt, bn).pvalue if n >= 6 else np.nan
        pw05 = TTestPower().power(effect_size=0.5, nobs=n, alpha=0.05)
        pw08 = TTestPower().power(effect_size=0.8, nobs=n, alpha=0.05)
        n_req05 = TTestPower().solve_power(effect_size=0.5, power=0.8, alpha=0.05)
        n_req08 = TTestPower().solve_power(effect_size=0.8, power=0.8, alpha=0.05)

        rows.append(dict(dataset=gse, n_pairs=n, scale_detected=scale,
                         median_burden_tumor=bt.mean(), median_burden_normal=bn.mean(),
                         paired_diff_mean=diff.mean(), ci95_lo=ci[0], ci95_hi=ci[1],
                         cohens_dz=dz, paired_t_p=t_p, wilcoxon_p=w_p,
                         power_at_dz0_5=pw05, power_at_dz0_8=pw08))
        power_lines.append(
            f"{gse}: n_pairs={n}, scale={scale}\n"
            f"  paired diff (tumor-normal) mean={diff.mean():.4f}, 95%CI [{ci[0]:.4f}, {ci[1]:.4f}], dz={dz:.3f}\n"
            f"  paired t p={t_p:.4f}, Wilcoxon p={w_p if not np.isnan(w_p) else float('nan'):.4f}\n"
            f"  power at dz=0.5: {pw05:.3f}; at dz=0.8: {pw08:.3f}\n"
            f"  n required for 80% power: dz=0.5 -> {n_req05:.1f} pairs; dz=0.8 -> {n_req08:.1f} pairs\n"
        )
        fig_data[gse] = (bn, bt, diff.mean(), ci)

    summ = pd.DataFrame(rows)
    summ.to_csv(f"{OUT}/rev1b_circrna_pilot.csv", index=False)
    with open(f"{OUT}/rev1b_power.txt", "w") as f:
        f.write("\n".join(power_lines))
        f.write("\nNote: both datasets use Arraystar circRNA microarrays (ASCRP probe IDs) that profile "
                "circRNA only (no mRNA channel); per-sample circRBP mRNA is not measurable in these "
                "datasets, so a circRBP-circRNA correlation cannot be computed from them.\n")
    print(summ.to_string(index=False))
    print("\n".join(power_lines))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, (gse, (bn, bt, dmean, ci)) in zip(axes, fig_data.items()):
        bp = ax.boxplot([bn, bt], tick_labels=["Non-tumor", "HCC"], patch_artist=True, widths=0.5,
                        medianprops=dict(color="black"))
        for p, c in zip(bp["boxes"], [COL_SAFE, COL_RISK]):
            p.set_facecolor(c); p.set_alpha(0.85)
        for i in range(len(bt)):
            ax.plot([1, 2], [bn[i], bt[i]], color="grey", lw=0.7, alpha=0.6)
        ax.errorbar([1.62], [bn.mean() + dmean], yerr=[[dmean - ci[0]], [ci[1] - dmean]],
                    fmt="none", ecolor="black", capsize=4, lw=1.2)
        ax.set_title(f"{gse} (median burden, paired)", fontsize=10)
        ax.set_ylabel("Median log2 circRNA intensity (all probes)")
    fig.suptitle("Global circRNA abundance in HCC vs matched non-tumor liver (pilot, exploratory)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/FigS10_circrna_burden.pdf", dpi=FIG_DPI); plt.close(fig)
    print("\n11a 完成")


if __name__ == "__main__":
    main()
