# circRBP-HCC：circRNA 生物发生相关 RNA 结合蛋白在肝细胞癌中的系统刻画与四基因预后签名

本仓库包含论文 *Systematic characterization of circRNA biogenesis-related RNA-binding proteins in hepatocellular carcinoma identifies a four-gene prognostic signature associated with dedifferentiation, immune evasion, and therapeutic vulnerability* 的完整分析代码。

作者：Wanli He¹\*、Weiwei Jiang¹（¹ School of Medicine, Shangqiu Institute of Technology, Shangqiu, Henan, China；\* 通讯作者 1350012019@sqgxy.edu.cn）

## 环境

- Python ≥ 3.10
- 依赖：`pandas`、`numpy`、`scipy`、`scikit-learn`、`scikit-survival`、`lifelines`、`gseapy`、`openpyxl`、`matplotlib`、`seaborn`

```bash
pip install pandas numpy scipy scikit-learn scikit-survival lifelines gseapy openpyxl matplotlib seaborn
```

## 数据获取（均为公开数据，请自行下载后放入 `rawdata/` 目录）

| 数据 | 来源 | 说明 |
|---|---|---|
| TCGA-LIHC 表达 | UCSC Xena (`TCGA-LIHC.star_fpkm.tsv.gz`) | log2(FPKM+1) |
| TCGA-LIHC 生存/临床 | UCSC Xena (`TCGA-LIHC.survival.tsv.gz`, `TCGA-LIHC.clinical.tsv.gz`) | OS 单位为天 |
| GSE14520 | GEO (GPL571/GPL3921 series matrix + 作者补充表) | 外部验证（OS/RFS） |
| GSE76427 | GEO (series matrix) | 外部验证（OS） |
| ICGC LIRI-JP | HCCDB（数据集 HCCDB18） | 第三外部验证队列（OS） |
| GSE94508 / GSE97332 | GEO（circRNA 芯片 series matrix） | circRNA 产出交叉核查 |
| GSE149614 | GEO（计数矩阵 + 官方标准化矩阵 + 元数据） | 单细胞验证（71,915 细胞） |
| GDSC1 药敏 | CancerRxGene (fitted dose response, 27Oct23) | 药物敏感性 |
| 细胞系表达 | Cell Model Passports（合并 RNA-seq）或 GDSC RMA basal expression | 药敏分析用 |
| HALLMARK 基因集 | MSigDB v7.5 (`h.all.v7.5.1.symbols.gmt`，本仓库已附) | ssGSEA / 增殖元基因 |

## 分析流程（按编号顺序运行）

| 脚本 | 内容 | 主要输出 |
|---|---|---|
| `01_data_prep.py` | 数据读取与预处理 | `results/tcga_expr.csv`、`results/*_clin*.csv` 等 |
| `02_screen_deg.py` | 差异表达与单因素 Cox 筛选 | `results/circrbp_deg.csv`、`results/univ_cox.csv` |
| `03_model.py` | LASSO-Cox 建模与验证（核心模型） | `results/model_genes.csv`、`results/risk_scores.csv`、`results/model_validation.csv`、Fig2 |
| `03b_function.py` | GO/KEGG/STRING 功能分析 | `results/go_enrich.csv`、`kegg_enrich.csv`、`string_network.csv` |
| `03c_hallmark.py` | Hallmark ssGSEA 与相关性 | `results/hallmark_ssgsea.csv`、`hallmark_corr.csv` |
| `04_clinical.py` | 独立预后因素、列线图、校准（TCGA） | Fig4 |
| `05_immune.py` | 免疫浸润与检查点分析 | `results/immune_ssgsea.csv`、Fig5 |
| `06_drug.py` | GDSC1 泛癌细胞系药敏关联 | `results/drug_spearman.csv`、Fig6 |
| `07_singlecell.py` | 单细胞表达验证（GSE149614） | `results/sc_*`、Fig7 |
| `08a/08b` | 官方标准化数据交叉验证与统计量核对 | 核对用 |
| `09a_lasso_stability.py` | bootstrap 稳定性、3 基因 vs 4 基因对比 | `results/rev3_*` |
| `09b_ph_and_score.py` | Schoenfeld PH 检验、评分分布、中位随访 | `results/rev4_*` |
| `09c_gse14520_calibration.py` | 外部队列（GSE14520）校准曲线 | `results/rev5_*` |
| `09d_hcc_drug_subset.py` | 肝癌细胞系子集药敏 | `results/rev6_*` |
| `09e_sc_effectsize.py` | 单细胞效应量（中位数/倍数变化/Cliff's δ） | `results/rev7_*` |
| `09f_bench_proliferation.py` | 与 G2M 增殖元基因基准比较 | `results/rev8_*` |
| `10a_icgc_lirijp_validation.py` | ICGC LIRI-JP（HCCDB18）第三外部验证队列 | `results/icgc_lirijp_validation.csv` |
| `10b_circrna_crosscheck.py` | circRNA 芯片数据集（GSE94508/GSE97332）交叉核查 | `results/rev1_circrna_*` |

## 说明

- 跨队列风险评分：各队列内部独立 z 分数标准化后乘以 TCGA 拟合的 LASSO 系数，未做跨数据集批次整合。
- 随机种子已在脚本内固定，结果可复现。
- 数据文件较大未随仓库提供；按上表链接下载后放入 `rawdata/` 即可顺序重跑全流程。
