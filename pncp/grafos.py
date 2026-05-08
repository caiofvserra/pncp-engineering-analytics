"""
Grafos órgão↔fornecedor.

Constrói:
  1. Grafo bipartido (órgão ↔ fornecedor, peso = nº contratos)
  2. Projeção 1-mode em fornecedores (co-ocorrência em órgãos)
  3. Detecta comunidades via Louvain
  4. Métricas de centralidade (top-N nós)
  5. "Red flags": fornecedores com poucos órgãos + alto valor + objeto eng

Saídas em dados/grafos/.
"""

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pncp import config
from pncp.io_disco import ler_parquet, salvar_parquet, salvar_json
from pncp.ram import liberar, com_gc


def construir_bipartido(df):
    """Bipartido: nós tipo='orgao' ou tipo='fornecedor'."""
    G = nx.Graph()
    cols = {
        "orgao": _primeira_coluna(df, ("orgaoEntidade", "razaoSocialOrgao",
                                        "nomeOrgao", "nome_orgao")),
        "fornec": _primeira_coluna(df, ("razaoSocialFornecedor",
                                          "fornecedor", "nomeRazaoSocialFornecedor")),
    }
    if cols["orgao"] is None or cols["fornec"] is None:
        print("[grafos] colunas de órgão/fornecedor não encontradas")
        return G

    pares = (df[[cols["orgao"], cols["fornec"]]]
             .dropna().astype(str)
             .groupby([cols["orgao"], cols["fornec"]]).size())
    for (orgao, fornec), peso in pares.items():
        G.add_node(orgao, tipo="orgao")
        G.add_node(fornec, tipo="fornecedor")
        G.add_edge(orgao, fornec, peso=int(peso))
    return G


def _primeira_coluna(df, candidatos):
    for c in candidatos:
        if c in df.columns:
            return c
    return None


def projetar_fornecedores(G_bip):
    """Projeção em fornecedores: aresta se compartilham um órgão."""
    fornecedores = [n for n, d in G_bip.nodes(data=True) if d.get("tipo") == "fornecedor"]
    return nx.bipartite.weighted_projected_graph(G_bip, fornecedores)


def centralidade(G, top_n=20):
    """Top-N por degree e betweenness (betweenness é caro: amostra k=200)."""
    deg = pd.Series(dict(G.degree())).sort_values(ascending=False).head(top_n)
    if len(G) > 0:
        bw = nx.betweenness_centrality(G, k=min(200, len(G)), seed=config.SEED)
        bw = pd.Series(bw).sort_values(ascending=False).head(top_n)
    else:
        bw = pd.Series(dtype=float)
    return {"top_degree": deg.to_dict(), "top_betweenness": bw.to_dict()}


def pagerank(G, top_n=20):
    """
    PageRank no grafo bipartido (Cap. 9.5.2, Han/Kamber/Pei). Mede
    importância estrutural — fornecedores com PageRank alto têm muitos
    órgãos parceiros e/ou parceiros importantes. Útil para cruzar com
    'red flags': PageRank baixo + objeto eng + rotulo geral = suspeito.
    """
    pr = nx.pagerank(G, alpha=0.85, max_iter=200)
    return (pd.Series(pr).sort_values(ascending=False).head(top_n).to_dict())


def comunidades_louvain(G):
    """Comunidades via Louvain (python-louvain)."""
    try:
        import community as community_louvain
    except ImportError:
        print("[grafos] python-louvain não instalado")
        return {}
    part = community_louvain.best_partition(G, random_state=config.SEED)
    contagem = Counter(part.values())
    return {"n_comunidades": len(contagem),
            "tamanhos": dict(contagem),
            "particao": part}


def red_flags(df, G_bip, k_orgaos_max=2):
    """
    Fornecedores com poucos órgãos parceiros (≤k) + objeto sugere engenharia
    + estão em contratos rotulados como 'geral'. Sinaliza concentração suspeita.
    """
    col_fornec = _primeira_coluna(df, ("razaoSocialFornecedor", "fornecedor"))
    if col_fornec is None:
        return pd.DataFrame()
    grau = pd.Series(dict(G_bip.degree()))
    fornec_baixo = grau[grau <= k_orgaos_max].index

    sub = df[df[col_fornec].astype(str).isin(fornec_baixo)].copy()
    if "n_termos_eng" in sub.columns:
        sub = sub[(sub["n_termos_eng"] > 0) & (sub["rotulo"] == "geral")]
    return sub


def visualizar(G, top_n=50, nome="grafo.png"):
    """Plota subgrafo dos top-N nós por grau (PNG)."""
    if len(G) == 0:
        return None
    top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:top_n]
    H = G.subgraph([n for n, _ in top])
    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(H, seed=config.SEED, k=0.4)
    cores = ["tab:blue" if G.nodes[n].get("tipo") == "orgao" else "tab:orange"
              for n in H.nodes]
    nx.draw_networkx_nodes(H, pos, node_color=cores, node_size=80, ax=ax)
    nx.draw_networkx_edges(H, pos, alpha=0.3, ax=ax)
    ax.set_title(f"Top-{top_n} nós (azul=órgão, laranja=fornecedor)")
    ax.axis("off")
    saida = config.caminho(config.SUB_P7, nome)
    fig.tight_layout()
    fig.savefig(saida, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return saida


@com_gc
def executar(caminho_parquet=None):
    from pncp.ram import precisa_de
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")
    if not precisa_de(caminho_parquet, "grafos",
                       "rode pncp.coleta.coletar(...) primeiro"):
        return None
    df = ler_parquet(caminho_parquet)
    if df.empty:
        print("[grafos] parquet vazio — pulando")
        return None

    G_bip = construir_bipartido(df)
    if len(G_bip) == 0:
        print("[grafos] grafo vazio — colunas de órgão/fornecedor faltando")
        return None

    resumo = {
        "n_nos": G_bip.number_of_nodes(),
        "n_arestas": G_bip.number_of_edges(),
        "n_orgaos": sum(1 for _, d in G_bip.nodes(data=True) if d.get("tipo") == "orgao"),
        "n_fornecedores": sum(1 for _, d in G_bip.nodes(data=True)
                                if d.get("tipo") == "fornecedor"),
        "centralidade": centralidade(G_bip),
    }

    # Projeção (cara em N grande — só roda se < 5k fornecedores)
    if resumo["n_fornecedores"] < 5000:
        G_proj = projetar_fornecedores(G_bip)
        resumo["comunidades_proj"] = comunidades_louvain(G_proj)
        if "particao" in resumo["comunidades_proj"]:
            resumo["comunidades_proj"].pop("particao")  # muito grande p/ JSON

    flags = red_flags(df, G_bip)
    if not flags.empty:
        salvar_parquet(flags, config.caminho(config.SUB_P7, "red_flags.parquet"))
        resumo["red_flags_count"] = int(len(flags))

    visualizar(G_bip, top_n=50, nome="bipartido_top50.png")
    salvar_json(resumo, config.caminho(config.SUB_P7, "resumo.json"))
    print(f"[grafos] {resumo['n_orgaos']} órgãos × "
          f"{resumo['n_fornecedores']} fornecedores")
    liberar(df, G_bip)
    return resumo
