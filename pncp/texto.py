"""
Pré-processamento de texto e construção da matriz TF-IDF.

A matriz TF-IDF é sempre sparse e salva em `.npz` — nunca densa em RAM.
Para 300k linhas × 50k features, sparse pesa ~30-50MB; densa pesaria ~60GB.
"""

import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from pncp import config
from pncp.io_disco import (
    ler_parquet, salvar_parquet, salvar_sparse, ler_sparse, salvar_modelo,
)
from pncp.ram import liberar


# ── Stopwords PT-BR (subset enxuto, evita dependência do NLTK) ───────────────
STOPWORDS_PT = frozenset("""
a o e de da do das dos para por com sem em no na nos nas
um uma uns umas que se sua seu suas seus
ao aos à às pela pelo pelas pelos
ser estar ter haver há foi era são é como mas ou também já não nao sim
este esta esse essa aquele aquela isso aquilo isto
""".split())


def _remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


_RX_NAO_ALFANUM = re.compile(r"[^a-z0-9\s]+")
_RX_MULTI_ESPACO = re.compile(r"\s+")


def limpar(texto) -> str:
    """Normaliza um texto: minúscula, sem acento, só alfanumérico, sem stopword."""
    if not isinstance(texto, str) or not texto:
        return ""
    t = _remover_acentos(texto.lower())
    t = _RX_NAO_ALFANUM.sub(" ", t)
    t = _RX_MULTI_ESPACO.sub(" ", t).strip()
    tokens = [w for w in t.split() if w not in STOPWORDS_PT and len(w) > 2]
    return " ".join(tokens)


def preprocessar(caminho_parquet):
    """
    Adiciona coluna `objeto_limpo` ao parquet e regrava.
    Retorna o mesmo path (para encadear).
    """
    df = ler_parquet(caminho_parquet)
    if "objeto_limpo" not in df.columns:
        df["objeto_limpo"] = df["objeto"].fillna("").map(limpar)
        salvar_parquet(df, caminho_parquet)
    liberar(df)
    return caminho_parquet


def construir_tfidf(caminho_parquet, caminho_saida=None):
    """
    Constrói TF-IDF (1,2-grams) sobre `objeto_limpo` e salva sparse + vectorizer.

    Returns:
        dict com paths: {"X": ..., "vec": ..., "labels": ...}
    """
    if caminho_saida is None:
        caminho_saida = config.caminho(config.SUB_P2)

    df = ler_parquet(caminho_parquet, colunas=["objeto_limpo", "rotulo"])
    vec = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        min_df=config.TFIDF_MIN_DF,
        ngram_range=config.TFIDF_NGRAM,
        sublinear_tf=True,
    )
    X = vec.fit_transform(df["objeto_limpo"].fillna(""))
    print(f"[texto] TF-IDF: {X.shape[0]:,} docs × {X.shape[1]:,} features "
          f"(nnz={X.nnz:,})")

    paths = {
        "X": salvar_sparse(X, caminho_saida / "X.npz"),
        "vec": salvar_modelo(vec, caminho_saida / "vectorizer.joblib"),
        "labels": salvar_parquet(df[["rotulo"]], caminho_saida / "labels.parquet"),
    }
    liberar(df, X, vec)
    return paths


def carregar_tfidf(caminho_saida=None):
    """Recarrega artefatos de TF-IDF do disco."""
    from pncp.io_disco import ler_modelo
    if caminho_saida is None:
        caminho_saida = config.caminho(config.SUB_P2)
    return {
        "X": ler_sparse(caminho_saida / "X.npz"),
        "vec": ler_modelo(caminho_saida / "vectorizer.joblib"),
        "labels": ler_parquet(caminho_saida / "labels.parquet"),
    }


def selecao_chi2(top_k=10_000, caminho_saida=None):
    """
    Reduz o vocabulário TF-IDF para os top-K termos com maior chi² em
    relação ao rótulo. Inspirado no Cap. 6 (Aggarwal & Zhai) — feature
    selection de filtro para texto.

    Útil quando max_features=50k está estourando RAM; chi² escolhe os
    termos mais discriminativos antes de descartar o resto.
    """
    from sklearn.feature_selection import SelectKBest, chi2

    if caminho_saida is None:
        caminho_saida = config.caminho(config.SUB_P2)
    art = carregar_tfidf()
    X, y = art["X"], art["labels"]["rotulo"].astype(str).values

    seletor = SelectKBest(chi2, k=min(top_k, X.shape[1]))
    X_red = seletor.fit_transform(X, y)
    print(f"[texto] chi²: {X.shape[1]:,} → {X_red.shape[1]:,} features")

    from pncp.io_disco import salvar_sparse, salvar_modelo
    salvar_sparse(X_red, caminho_saida / "X_chi2.npz")
    salvar_modelo(seletor, caminho_saida / "seletor_chi2.joblib")
    liberar(X, X_red, art)
    return caminho_saida


def marcar_termos_dominio(caminho_parquet):
    """
    Conta ocorrências dos termos de engenharia em cada objeto.
    Usado pela EDA e como feature complementar à TF-IDF.
    """
    df = ler_parquet(caminho_parquet)
    if "objeto_limpo" not in df.columns:
        df["objeto_limpo"] = df["objeto"].fillna("").map(limpar)

    termos = [_remover_acentos(t.lower()) for t in config.TERMOS_ENGENHARIA]
    contagem = np.zeros(len(df), dtype="int16")
    for termo in termos:
        contagem += df["objeto_limpo"].str.contains(termo, regex=False, na=False) \
                    .astype("int16")
    df["n_termos_eng"] = contagem
    salvar_parquet(df, caminho_parquet)
    liberar(df)
    return caminho_parquet
