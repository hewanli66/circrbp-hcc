# -*- coding: utf-8 -*-
"""
03b 功能刻画（新框架 Fig3）
  - GO / KEGG 富集分析（59 个差异 circRBP，Enrichr 在线接口）
  - STRING PPI 网络（20 个预后相关 circRBP）
输出：
  figs/Fig3A_go.pdf, Fig3B_kegg.pdf, Fig3C_ppi.pdf
  results/go_enrich.csv, results/kegg_enrich.csv, results/string_network.csv
"""
import sys, os, io
import pandas as pd, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUT, FIG
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"


def enrichr(genes, library):
    import gseapy as gp
    enr = gp.enrichr(gene_list=genes, gene_sets=library, organism="human",
                     outdir=None, no_plot=True, verbose=False)
    df = enr.results.sort_values("Adjusted P-value")
    df = df.rename(columns={"Term": "term", "Adjusted P-value": "adj_pvalue",
                            "P-value": "pvalue", "Combined Score": "combined_score"})
    return df


def barplot(df, title, path, topn=12):
    d = df.head(topn).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    neglog = -np.log10(d["adj_pvalue"].clip(lower=1e-300))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(d)))
    ax.barh(d["term"], neglog, color=colors)
    ax.set_xlabel("-log10(adjusted p value)")
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def string_ppi(genes, out_tsv):
    import urllib.request, urllib.parse
    params = urllib.parse.urlencode({
        "identifiers": "\n".join(genes), "species": 9606,
        "required_score": 400, "network_type": "functional"})
    url = f"https://string-db.org/api/tsv/network?{params}"
    with urllib.request.urlopen(url, timeout=90) as r:
        txt = r.read().decode()
    df = pd.read_csv(io.StringIO(txt), sep="\t")
    df.to_csv(out_tsv, index=False)
    return df


def ppi_plot(net, genes, path):
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(genes)
    for _, r in net.iterrows():
        if r["preferredName_A"] in genes and r["preferredName_B"] in genes:
            g.add_edge(r["preferredName_A"], r["preferredName_B"], score=float(r["score"]))
    deg = dict(g.degree)
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(g, seed=42, k=1.2)
    size = [300 + 200 * deg.get(n, 0) for n in g.nodes]
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=.4, width=[g[u][v]["score"] * 2 for u, v in g.edges])
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=size, node_color="#4DBBD5", alpha=.85, edgecolors="k")
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8)
    ax.set_title("PPI network of prognostic circRBPs (STRING)")
    ax.axis("off"); fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    print(f"PPI: {g.number_of_nodes()} 节点, {g.number_of_edges()} 边")


def main():
    deg = pd.read_csv(f"{OUT}/circrbp_deg.csv")
    genes = deg[deg.FDR < 0.05].gene.tolist()
    print(f"富集分析输入: {len(genes)} 个差异 circRBP")

    go = enrichr(genes, "GO_Biological_Process_2023")
    go.to_csv(f"{OUT}/go_enrich.csv", index=False)
    barplot(go, "GO Biological Process enrichment", f"{FIG}/Fig3A_go.pdf")

    kg = enrichr(genes, "KEGG_2021_Human")
    kg.to_csv(f"{OUT}/kegg_enrich.csv", index=False)
    barplot(kg, "KEGG pathway enrichment", f"{FIG}/Fig3B_kegg.pdf")
    print("KEGG 前5:", kg.head(5)["term"].tolist())

    prognostic = pd.read_csv(f"{OUT}/univ_cox.csv").gene.tolist()
    net = string_ppi(prognostic, f"{OUT}/string_network.csv")
    ppi_plot(net, prognostic, f"{FIG}/Fig3C_ppi.pdf")
    print("03b 完成")


if __name__ == "__main__":
    main()
