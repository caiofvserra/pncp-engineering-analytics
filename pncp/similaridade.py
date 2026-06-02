"""
Busca semântica de contratos similares — RAG para investigação.

Dado um contrato suspeito confirmado, encontra outros parecidos no
corpus que merecem o mesmo escrutínio usando embeddings SBERT.

Útil para o jurista: "achei 1 caso de subenquadramento — quais outros
são parecidos?". A LLM/auditor sabe que se o padrão repete em N órgãos
diferentes, vale fazer auto regulatório.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pncp import config
from pncp.io_disco import ler_parquet, salvar_parquet, ler_npy
from pncp.ram import com_gc


def _carregar_embeddings():
    """Carrega embeddings densos (gerados por pncp.embeddings.gerar)."""
    saida = config.caminho(config.SUB_EMB)
    arq = saida / "emb_sbert.npy"
    lbl = saida / "emb_sbert_labels.parquet"
    if not arq.exists() or not lbl.exists():
        return None, None
    return ler_npy(arq, mmap=False).astype("float32"), ler_parquet(lbl)


def buscar_similares(num_controle_pncp, k=20, modelo="sentence-bert"):
    """
    Para um contrato dado, retorna os top-K mais similares por cosseno
    no espaço de embeddings (estilo `util.cos_sim` da SBERT).

    Use depois de pncp.embeddings.gerar(). Se quiser buscar por texto
    livre (não NCP), use `buscar_por_texto()` abaixo.
    """
    emb, labels = _carregar_embeddings()
    if emb is None:
        print("[similaridade] rode pncp.embeddings.gerar() primeiro")
        return None

    # Localiza o contrato pelo NCP no DataFrame de coleta
    df = ler_parquet(config.caminho(config.SUB_COLETA, "contratos.parquet"),
                     colunas=["numeroControlePNCP", "objeto", "rotulo",
                              "valor", "razaoSocialOrgao",
                              "nomeRazaoSocialFornecedor"])
    if "numeroControlePNCP" not in df.columns:
        return None

    # Alinha embeddings com df (assume mesma ordem na coleta)
    n = min(len(df), len(emb))
    df = df.head(n).reset_index(drop=True)
    emb = emb[:n]

    mask = df["numeroControlePNCP"] == num_controle_pncp
    if not mask.any():
        print(f"[similaridade] contrato {num_controle_pncp} não encontrado")
        return None

    idx_alvo = int(mask.idxmax())
    q = emb[idx_alvo:idx_alvo + 1]

    # Cosine via normalização + dot product (vetores já vêm normalizados
    # por SBERT, mas garantimos)
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    scores = (emb_norm @ q_norm.T).ravel()

    top = np.argsort(scores)[::-1][:k + 1]
    out = df.iloc[top].copy()
    out["similaridade"] = scores[top]
    # Remove o próprio alvo (similaridade ~1.0)
    out = out[out["numeroControlePNCP"] != num_controle_pncp].head(k)
    return out.reset_index(drop=True)


def buscar_por_texto(texto, k=20):
    """
    Encontra contratos similares a um TEXTO arbitrário (não precisa
    existir no PNCP). Útil para juristas: "tenho um achado, busco
    contratos parecidos."

    Exemplo: buscar_por_texto("manutenção elétrica com substituição
    de quadros de distribuição e ART do CREA")
    """
    emb, labels = _carregar_embeddings()
    if emb is None:
        print("[similaridade] rode pncp.embeddings.gerar() primeiro")
        return None

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[similaridade] instale sentence-transformers")
        return None
    modelo = SentenceTransformer(config.MODELO_SBERT)
    q = modelo.encode([texto], normalize_embeddings=True).astype("float32")

    df = ler_parquet(config.caminho(config.SUB_COLETA, "contratos.parquet"),
                     colunas=["numeroControlePNCP", "objeto", "rotulo",
                              "valor", "razaoSocialOrgao"])
    n = min(len(df), len(emb))
    df = df.head(n).reset_index(drop=True)
    emb = emb[:n]
    scores = (emb @ q.T).ravel()
    top = np.argsort(scores)[::-1][:k]
    out = df.iloc[top].copy()
    out["similaridade"] = scores[top]
    return out.reset_index(drop=True)


@com_gc
def expandir_suspeitos(top_n=20, k_por_suspeito=10):
    """
    Para cada um dos top-N suspeitos consolidados, busca os k contratos
    mais similares no corpus. Gera CSV consolidado com TODOS — útil para
    o jurista ter uma lista grande de potenciais para investigar.
    """
    susp_path = config.caminho(config.SUB_P9, "suspeitos_consolidados.parquet")
    if not Path(susp_path).exists():
        print("[similaridade] rode pncp.relatorio.gerar() primeiro")
        return None
    suspeitos = ler_parquet(susp_path).head(top_n)

    todos_relacionados = []
    for _, row in suspeitos.iterrows():
        ncp = row.get("numeroControlePNCP")
        if not ncp:
            continue
        sim = buscar_similares(ncp, k=k_por_suspeito)
        if sim is None or sim.empty:
            continue
        sim["seed_ncp"] = ncp
        sim["seed_objeto"] = str(row.get("objeto", ""))[:100]
        todos_relacionados.append(sim)

    if not todos_relacionados:
        return None
    consolidado = pd.concat(todos_relacionados, ignore_index=True)
    saida = config.caminho("similaridade", "expansao_suspeitos.csv")
    consolidado.to_csv(saida, index=False, encoding="utf-8-sig")
    print(f"[similaridade] {len(consolidado)} contratos relacionados a "
          f"{len(suspeitos)} suspeitos seed → {saida}")
    return consolidado
