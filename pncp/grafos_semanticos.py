"""
Grafo semântico de contratos via k-NN sobre embeddings.

Baseado nos notebooks 03 (BOW + k-NN), 09 P3 (Label Propagation),
12 P2 (LLM gera indicadores por cluster) e 18 (Agentes) do MBA.

Diferente de pncp.grafos (que é bipartido órgão↔fornecedor), aqui o
grafo conecta CONTRATOS por similaridade semântica. Comunidades
detectadas = padrões repetidos de subenquadramento.

Útil para:
  - Detectar que vários órgãos diferentes têm o MESMO tipo de
    subenquadramento (sinal de padrão sistêmico)
  - Propagar rótulos: dado um suspeito confirmado, sklearn-style
    label propagation marca os vizinhos semânticos
  - LLM gera indicador-síntese por cluster
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pncp import config
from pncp._plot import salvar_e_mostrar
from pncp.io_disco import ler_parquet, salvar_parquet, ler_npy, salvar_json
from pncp.ram import com_gc


def _embeddings_e_meta():
    """Carrega embeddings + metadata alinhados."""
    saida_emb = config.caminho(config.SUB_EMB)
    arq_emb = saida_emb / "emb_sbert.npy"
    if not arq_emb.exists():
        print("[grafos_sem] rode pncp.embeddings.gerar() primeiro")
        return None, None
    emb = ler_npy(arq_emb, mmap=False).astype("float32")
    # Normaliza p/ cosseno
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    df = ler_parquet(config.caminho(config.SUB_COLETA, "contratos.parquet"),
                     colunas=["numeroControlePNCP", "objeto", "rotulo",
                              "valor", "razaoSocialOrgao"])
    n = min(len(df), len(emb))
    return emb[:n], df.head(n).reset_index(drop=True)


@com_gc
def construir(k=5, focar_em="geral", limite=10_000):
    """
    Constrói grafo k-NN sobre embeddings dos contratos.

    Args:
      k: nº de vizinhos por contrato (5 ou 7 é típico)
      focar_em: rótulo a filtrar (None = todos). Default 'geral' — onde
        está o subenquadramento. Cluster de 'geral' próximos pode revelar
        padrão sistêmico.
      limite: corta para os primeiros N (memória). k-NN é O(n²) em RAM.
    """
    import networkx as nx
    from sklearn.neighbors import kneighbors_graph
    from networkx.algorithms import community

    emb, df = _embeddings_e_meta()
    if emb is None:
        return None
    if focar_em:
        mask = df["rotulo"] == focar_em
        df = df[mask].reset_index(drop=True)
        emb = emb[mask.values]
    if limite and len(df) > limite:
        df = df.head(limite)
        emb = emb[:limite]
    if len(df) < k + 1:
        print(f"[grafos_sem] amostra muito pequena: {len(df)}")
        return None

    print(f"[grafos_sem] {len(df):,} contratos | k={k}")
    A = kneighbors_graph(emb, n_neighbors=k, metric="cosine",
                          include_self=False)
    G = nx.Graph(A)

    # Comunidades via label propagation (Raghavan et al. 2007).
    # Diferente do Louvain de pncp.grafos — operação local, rápida em
    # grafos esparsos como k-NN.
    print("[grafos_sem] detectando comunidades (label propagation)...")
    cluster_id = 0
    for cluster_nodes in community.label_propagation_communities(G):
        for n in cluster_nodes:
            G.nodes[n]["cluster"] = cluster_id
        cluster_id += 1

    df["cluster"] = [G.nodes[i].get("cluster", -1) for i in range(len(df))]
    n_clusters = df["cluster"].nunique()
    sizes = df["cluster"].value_counts()
    print(f"[grafos_sem] {n_clusters} comunidades detectadas. "
          f"Maiores: {dict(sizes.head(5))}")

    # Persiste
    saida = config.caminho("grafos_semanticos")
    salvar_parquet(df, saida / "clusters_contratos.parquet")
    salvar_json({
        "n_contratos": int(len(df)),
        "n_clusters": int(n_clusters),
        "tamanho_top_10": sizes.head(10).to_dict(),
        "k": k,
        "focar_em": focar_em,
    }, saida / "resumo.json")
    return df


def amostrar_por_cluster(df_clusters=None, n_por_cluster=5, min_tamanho=10):
    """
    Pega uma amostra de cada cluster grande. Útil para passar ao LLM
    gerar indicadores (não dá pra mandar 10k contratos).
    """
    if df_clusters is None:
        p = config.caminho("grafos_semanticos", "clusters_contratos.parquet")
        if not Path(p).exists():
            print("[grafos_sem] rode construir() primeiro")
            return None
        df_clusters = ler_parquet(p)
    sizes = df_clusters["cluster"].value_counts()
    grandes = sizes[sizes >= min_tamanho].index.tolist()
    amostras = []
    for c in grandes:
        sub = df_clusters[df_clusters["cluster"] == c].head(n_por_cluster).copy()
        sub["_tamanho_cluster"] = int(sizes[c])
        amostras.append(sub)
    if not amostras:
        return pd.DataFrame()
    return pd.concat(amostras, ignore_index=True)


def propagar_rotulos(seeds_subenq, df_clusters=None):
    """
    Dada uma lista de NCPs confirmados como subenquadramento, propaga
    rótulo via cluster: todo contrato no MESMO cluster fica como candidato.

    Notebook 09 P3 (simple regularization). Implementação simplificada
    por cluster (vs random walk completo).
    """
    if df_clusters is None:
        p = config.caminho("grafos_semanticos", "clusters_contratos.parquet")
        if not Path(p).exists():
            return None
        df_clusters = ler_parquet(p)

    seeds = set(seeds_subenq)
    df = df_clusters.copy()
    df["seed_confirmado"] = df["numeroControlePNCP"].isin(seeds)

    # Clusters com ≥1 seed: todos os contratos do cluster são candidatos
    clusters_com_seed = df[df["seed_confirmado"]]["cluster"].unique()
    df["candidato_propagado"] = (df["cluster"].isin(clusters_com_seed) &
                                   ~df["seed_confirmado"])

    n_seeds = int(df["seed_confirmado"].sum())
    n_propagados = int(df["candidato_propagado"].sum())
    print(f"[propagar] {n_seeds} seeds → {n_propagados} candidatos via "
          f"{len(clusters_com_seed)} cluster(s)")

    saida = config.caminho("grafos_semanticos",
                            "propagacao_rotulos.parquet")
    salvar_parquet(df, saida)
    # CSV dos candidatos para fácil revisão
    df[df["candidato_propagado"]].to_csv(
        config.caminho("grafos_semanticos",
                        "candidatos_por_propagacao.csv"),
        index=False, encoding="utf-8-sig")
    return df
