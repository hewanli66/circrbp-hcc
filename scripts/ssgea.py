# -*- coding: utf-8 -*-
"""ssGSEA 公共工具函数"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def ssgsea_from_gmt(expr, gmt):
    """简化 ssGSEA：expr 行=基因、列=样本；gmt 为 {名称: [基因]}；返回 通路x样本 评分"""
    E = expr.values
    idx_map = {g: i for i, g in enumerate(expr.index)}
    n = E.shape[0]
    ranks = np.apply_along_axis(rankdata, 0, E)
    out = {}
    for name, genes in gmt.items():
        present = [g for g in genes if g in idx_map]
        if len(present) < 3:
            continue
        P = ranks[[idx_map[g] for g in present], :]
        score = (np.abs(P - (n + 1) / 2)).sum(axis=0) / len(present)
        out[name] = score
    return pd.DataFrame(out, index=expr.columns).T
