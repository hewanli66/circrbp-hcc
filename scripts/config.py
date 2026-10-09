# -*- coding: utf-8 -*-
"""
全局配置文件：路径、阈值、基因集
项目：circRNA生物发生相关RNA结合蛋白（circRBP）在肝细胞癌中的预后模型
"""
import os

# ---------- 目录 ----------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "rawdata")          # 用户手动下载的原始数据
OUT = os.path.join(BASE, "results")          # 中间结果（表达矩阵、临床表等）
FIG = os.path.join(BASE, "results", "figs")  # 论文图片
GENESET = os.path.join(BASE, "scripts", "genesets")
for d in (RAW, OUT, FIG, GENESET):
    os.makedirs(d, exist_ok=True)

# ---------- 原始文件（用户放入 rawdata 后自动识别） ----------
TCGA_EXPR   = os.path.join(RAW, "TCGA-LIHC.star_fpkm.tsv.gz")          # Xena，已为 log2(FPKM+1)
TCGA_SURV   = os.path.join(RAW, "TCGA-LIHC.survival.tsv.gz")           # Xena，OS.time 单位为天
TCGA_PHENO  = os.path.join(RAW, "TCGA-LIHC.clinical.tsv.gz")           # Xena
GSE14520_A  = os.path.join(RAW, "GSE14520-GPL571_series_matrix.txt.gz")   # GEO 表达
GSE14520_B  = os.path.join(RAW, "GSE14520-GPL3921_series_matrix.txt.gz")  # GEO 表达（备用）
GSE14520_SUP= os.path.join(RAW, "GSE14520_Extra_Supplement.txt.gz")       # GEO 生存补充
GSE76427    = os.path.join(RAW, "GSE76427_series_matrix.txt.gz")          # GEO
GDSC_IC50   = os.path.join(RAW, "GDSC1_fitted_dose_response_27Oct23.xlsx")  # GDSC（未下载）
GDSC_EXPR   = os.path.join(RAW, "GDSC1_gene_expression.xlsx")          # GDSC（如需）

# ---------- 统计阈值 ----------
FDR = 0.05          # 差异表达 FDR 阈值
LOGFC = 1.0         # 差异表达 |log2FC| 阈值
COX_P = 0.05        # 单因素 Cox 筛选 p 阈值
OS_MIN_DAYS = 30    # 最短随访天数

# ---------- 基因集 ----------
CIRCRBP_LIST = os.path.join(GENESET, "circrbp_genes.txt")  # circRNA生物发生相关RBP列表
# 来源：Li X et al., Mol Cell 2017, 67(2):214-227（103个RBP的功能缺失筛选）
#       叠加文献补充（Conn 2015 QKI; Fei 2017 HNRNPL; Ivanov 2015 ADAR; Errichelli 2017 FUS 等）

# ---------- 图形风格 ----------
FIG_DPI = 300
COL_RISK = "#E64B35"   # 高风险颜色（ Nature 风格红）
COL_SAFE = "#4DBBD5"   # 低风险颜色（蓝）
COL_GROUP = ["#4DBBD5", "#E64B35"]
