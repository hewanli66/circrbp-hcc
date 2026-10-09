# -*- coding: utf-8 -*-
"""
01 数据整理（适配实际下载文件）
输入：rawdata/ 中已核査的文件
  TCGA-LIHC.star_fpkm.tsv.gz   已是 log2(FPKM+1)，Ensembl ID
  TCGA-LIHC.survival.tsv.gz    OS.time 单位天
  TCGA-LIHC.clinical.tsv.gz    含 ajcc_pathologic_stage / age_at_index / gender
  GSE14520-GPL3921_series_matrix.txt.gz  主队列（445 样本与生存补充表全匹配）
  GSE14520_Extra_Supplement.txt.gz       生存与临床（按 Affy_GSM 关联）
  GSE76427_series_matrix.txt.gz  原始强度（需 log2），含 event_os / duryears_os
运行时会自动下载（如缓存中没有）：
  gencode.v22.annotation.gtf.gz   Ensembl->Symbol 映射（EBI, 41MB）
  GPL3921.annot.gz / GPL10558.annot.gz   探针->Symbol 映射（NCBI GEO, 共12MB）
输出：results/ 下的标准化表达矩阵与临床表
"""
import sys, os, gzip, urllib.request, urllib.parse, json
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RAW, OUT, TCGA_EXPR, TCGA_SURV, TCGA_PHENO, GSE14520_B, GSE14520_SUP, GSE76427, OS_MIN_DAYS

CACHE = os.path.join(RAW, "annotation_cache"); os.makedirs(CACHE, exist_ok=True)


def fetch(url, dest):
    if os.path.exists(dest): return dest
    print(f"下载注释文件: {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


# ---------- Ensembl -> Symbol ----------
def ensembl_to_symbol():
    gtf = fetch("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_22/gencode.v22.annotation.gtf.gz",
                os.path.join(CACHE, "gencode.v22.annotation.gtf.gz"))
    m = {}
    with gzip.open(gtf, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split("\t")
            if len(p) < 9 or p[2] != "gene": continue
            info = dict(x.strip().split(" ", 1) for x in p[8].rstrip(";").split("; ") if " " in x)
            gid = info.get("gene_id", "").strip('"').split(".")[0]
            gname = info.get("gene_name", "").strip('"')
            if gid: m[gid] = gname
    return m


# ---------- 探针 -> Symbol ----------
def probe_to_symbol_illumina():
    """GPL10558：解析本地 Illumina 官方 manifest（随 GSE76427_RAW.tar 提供），Probe_Id -> Symbol"""
    path = os.path.join(CACHE, "GPL10558_manifest.txt.gz")
    if not os.path.exists(path):
        raise FileNotFoundError("缺少 GPL10558_manifest.txt.gz（请先从 GSE76427_RAW.tar 解出，见 README）")
    ids, syms = [], []
    in_probes = False
    with gzip.open(path, "rt", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("[Probes]"):
                header = next(f).rstrip("\n").split("\t"); in_probes = True; continue
            if not in_probes or line.startswith("["):
                continue
            p = line.split("\t")
            if len(p) < 14: continue
            ids.append(p[13].strip())    # Probe_Id
            syms.append(p[11].strip())   # Symbol
    s = pd.Series(syms, index=ids)
    return s[(s != "")]


def probe_to_symbol_affy(probes):
    """GPL3921（Affy HG-U133A）：mygene.info 'reporter' 批量注释（带本地缓存）"""
    cache = os.path.join(CACHE, "gpl3921_probe_symbol.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache, index_col=0).iloc[:, 0]
        return df
    probes = [str(p) for p in probes]
    mapping = {}
    for i in range(0, len(probes), 800):
        batch = probes[i:i + 800]
        data = urllib.parse.urlencode({
            "q": ",".join(batch), "scopes": "reporter",
            "fields": "symbol", "species": "human"}).encode()
        req = urllib.request.Request("https://mygene.info/v3/query", data=data, headers={
            "User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"})
        with urllib.request.urlopen(req, timeout=60) as r:
            for hit in json.loads(r.read()):
                q = hit.get("query")
                if q not in mapping and "symbol" in hit:   # 取第一个命中
                    mapping[q] = hit["symbol"]
        print(f"  mygene 注释进度: {min(i+800, len(probes))}/{len(probes)}")
    s = pd.Series(mapping)
    s.to_csv(cache)
    return s


# ---------- TCGA ----------
def load_tcga():
    df = pd.read_csv(TCGA_EXPR, sep="\t", index_col=0, compression="gzip")
    df.index = [i.split(".")[0] for i in df.index]              # 去版本号
    m = ensembl_to_symbol()
    df["sym"] = pd.Series(df.index, index=df.index).map(m).values
    df = df.dropna(subset=["sym"]); df = df[df.sym != ""]
    df = df.groupby("sym").mean().T                              # 行=样本 列=基因
    df.index = [c[:15] for c in df.index]
    df = df[~df.index.duplicated()]
    expr = df.T                                                  # 行=基因 列=样本

    surv = pd.read_csv(TCGA_SURV, sep="\t", index_col=0, compression="gzip")
    surv.index = [s[:15] for s in surv.index]
    surv = surv[~surv.index.duplicated(keep="first")]
    clin = pd.DataFrame({"sample": expr.columns})
    clin["OS"] = clin["sample"].map(surv["OS"])
    clin["OS.time"] = clin["sample"].map(surv["OS.time"])

    ph = pd.read_csv(TCGA_PHENO, sep="\t", index_col=0, compression="gzip")
    ph.index = [s[:15] for s in ph.index]
    ph = ph[~ph.index.duplicated(keep="first")]
    clin["age"] = pd.to_numeric(clin["sample"].map(ph.get("age_at_index.demographic")), errors="coerce")
    clin["gender"] = clin["sample"].map(ph.get("gender.demographic"))
    stage = clin["sample"].map(ph.get("ajcc_pathologic_stage.diagnoses")).astype(str)
    clin["stage"] = stage.str.upper().str.extract(r"STAGE ([IV]+)")[0]

    clin = clin.dropna(subset=["OS", "OS.time"])
    clin = clin[clin["OS.time"] >= OS_MIN_DAYS]
    return expr[clin["sample"]], clin.reset_index(drop=True)


# ---------- GEO series matrix ----------
def load_geo(path):
    opener = gzip.open if path.endswith(".gz") else open
    chars, ids, rows, header = {}, [], [], None
    with opener(path, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                ids = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics"):
                for i, item in enumerate(line.rstrip("\n").split("\t")[1:]):
                    item = item.strip().strip('"')
                    if ":" in item:
                        k, v = item.split(":", 1)
                        chars.setdefault(k.strip(), {})[i] = v.strip()
            elif line.startswith("!series_matrix_table_begin"):
                header = [x.strip('"') for x in next(f).rstrip("\n").split("\t")]
                for line2 in f:
                    if line2.startswith("!series_matrix_table_end"): break
                    rows.append(line2.rstrip("\n").split("\t"))
    df = pd.DataFrame(rows, columns=header).set_index(header[0])
    df.index = [i.strip('"') for i in df.index]
    df = df.apply(pd.to_numeric, errors="coerce")
    clin = pd.DataFrame({"sample": ids})
    for k, d in chars.items():
        clin[k] = [d.get(i, np.nan) for i in range(len(ids))]
    return df, clin


def collapse_probe(expr, mapping):
    """探针->Symbol：多探针取均值，无注释丢弃"""
    e = expr.copy()
    e["sym"] = mapping.reindex(e.index)
    e = e.dropna(subset=["sym"])
    e = e[e.sym != ""]
    return e.groupby("sym").mean()


def load_gse14520():
    expr, _ = load_geo(GSE14520_B)
    sup = pd.read_csv(GSE14520_SUP, sep="\t", compression="gzip")
    sup.columns = [c.strip() for c in sup.columns]
    print("  生存补充表列名:", list(sup.columns))
    # 动态匹配生存时间/状态列
    tcol = [c for c in sup.columns if "survival" in c.lower() and ("time" in c.lower() or "month" in c.lower())]
    scol = [c for c in sup.columns if "survival" in c.lower() and ("status" in c.lower() or "death" in c.lower())]
    bcol = [c for c in sup.columns if "bclc" in c.lower()]
    sup = sup.rename(columns={tcol[0]: "OS.time", scol[0]: "OS"} if tcol and scol else {})
    sup["sample"] = sup["Affy_GSM"].astype(str).str.strip()
    keep = ["sample", "OS.time", "OS"] + ([bcol[0]] if bcol else [])
    clin = sup[keep].rename(columns={bcol[0]: "stage"} if bcol else {})
    clin["OS"] = pd.to_numeric(clin["OS"], errors="coerce")
    clin["OS.time"] = pd.to_numeric(clin["OS.time"], errors="coerce") * 30.44  # 月->天
    expr = collapse_probe(expr, probe_to_symbol_affy(expr.index))
    return expr, clin


def load_gse76427():
    expr, clin_raw = load_geo(GSE76427)
    expr = np.log2(expr + 1)
    c = pd.DataFrame({"sample": clin_raw["sample"]})
    c["OS"] = pd.to_numeric(clin_raw.get("event_os"), errors="coerce")
    c["OS.time"] = pd.to_numeric(clin_raw.get("duryears_os"), errors="coerce") * 365.25
    c["age"] = pd.to_numeric(clin_raw.get("age (years)"), errors="coerce")
    c["gender"] = clin_raw.get("gender (1=m, 2=f)")
    c["stage"] = clin_raw.get("bclc_staging")
    expr = collapse_probe(expr, probe_to_symbol_illumina())
    return expr, c


def main():
    expr, clin = load_tcga()
    expr.to_csv(f"{OUT}/tcga_expr.csv"); clin.to_csv(f"{OUT}/tcga_clin.csv", index=False)
    bar = pd.Series(expr.columns).str[13:15]
    print(f"TCGA: {expr.shape[1]} 样本（肿瘤{(bar=='01').sum()}，正常{(bar=='11').sum()}）x {expr.shape[0]} 基因；随访 {len(clin)}")

    e, c = load_gse14520()
    e.to_csv(f"{OUT}/gse14520_expr_symbol.csv"); c.to_csv(f"{OUT}/gse14520_clin_raw.csv", index=False)
    print(f"GSE14520: {e.shape[1]} 样本 x {e.shape[0]} 基因；临床表 {len(c)}")

    e, c = load_gse76427()
    e.to_csv(f"{OUT}/gse76427_expr_symbol.csv"); c.to_csv(f"{OUT}/gse76427_clin_raw.csv", index=False)
    tum = c["sample"].isin(e.columns)
    print(f"GSE76427: {e.shape[1]} 样本 x {e.shape[0]} 基因；含生存信息 {c.OS.notna().sum()}")
    print("01 完成")


if __name__ == "__main__":
    main()
