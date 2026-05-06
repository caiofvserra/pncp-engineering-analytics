"""
Detecção de outliers — contratos suspeitos como anomalias contextuais.

Inspirado no Cap. 11 de Han, Kamber & Pei (Data Mining: Concepts and
Techniques). A ideia: dentro do cluster "serviços gerais", a maioria
dos contratos é homogeneamente "geral" (limpeza, vigilância, alimentação).
Um contrato com vocabulário, valor, ou comportamento muito distinto pode
ser um outlier contextual — candidato a subenquadramento.

Métodos:
  - IsolationForest (global outliers, escalável a 300k)
  - LOF — Local Outlier Factor (contextual, mais caro)

Saídas em dados/outliers/:
  - scores.parquet (cada contrato com score de anomalia)
  - top_anomalos.parquet (top-N "geral" mais anômalos)
"""

import numpy as np
import pandas as pd

from pncp import config
from pncp.io_disco import (
    ler_parquet, salvar_parquet, salvar_json,
)
from pncp.ram import liberar, com_gc
from pncp.texto import carregar_tfidf


def _reduzir_dim(X, n=64):
    """Reduz TF-IDF esparso para denso baixo-dim antes do detector."""
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n, random_state=config.SEED)
    return svd.fit_transform(X).astype("float32")


@com_gc
def isolation_forest(contaminacao=0.05, n_estimators=200):
    """
    IsolationForest sobre TF-IDF reduzido. Score quanto mais negativo,
    mais anômalo. Escalável a 300k contratos.
    """
    from sklearn.ensemble import IsolationForest

    art = carregar_tfidf()
    X = _reduzir_dim(art["X"], n=64)
    rotulos = art["labels"]["rotulo"].astype(str).values

    # Treina apenas no cluster 'geral' — queremos outliers DENTRO desse cluster
    mask_geral = rotulos == "geral"
    if mask_geral.sum() < 100:
        print("[outliers] poucos 'geral' para detecção contextual")
        return None

    iso = IsolationForest(
        n_estimators=n_estimators, contamination=contaminacao,
        random_state=config.SEED, n_jobs=-1,
    )
    iso.fit(X[mask_geral])
    score = iso.score_samples(X)   # quanto menor, mais anômalo

    out = pd.DataFrame({
        "rotulo": rotulos,
        "score_anomalia_iforest": score.astype("float32"),
    })
    saida = config.caminho("outliers", "scores_iforest.parquet")
    salvar_parquet(out, saida)

    # Top-N anômalos dentro de 'geral' (mais negativos)
    top = (out[out["rotulo"] == "geral"]
           .sort_values("score_anomalia_iforest", ascending=True)
           .head(500))
    salvar_parquet(top, config.caminho("outliers", "top_anomalos_iforest.parquet"))

    salvar_json({
        "metodo": "IsolationForest",
        "contamination": contaminacao,
        "n_treino_geral": int(mask_geral.sum()),
        "n_top_salvos": int(len(top)),
    }, config.caminho("outliers", "resumo_iforest.json"))
    print(f"[outliers] IsolationForest → top {len(top)} anômalos em 'geral'")
    liberar(art, X, iso)
    return saida


@com_gc
def lof(n_neighbors=20, contaminacao=0.05, max_amostras=20000):
    """
    LOF: detecta outliers locais (contextuais). Mais caro que IsolationForest;
    aplica subsample para 300k linhas.
    """
    from sklearn.neighbors import LocalOutlierFactor

    art = carregar_tfidf()
    X = _reduzir_dim(art["X"], n=64)
    rotulos = art["labels"]["rotulo"].astype(str).values

    # Subsample (LOF é O(n²) na pior hipótese)
    if X.shape[0] > max_amostras:
        rng = np.random.default_rng(config.SEED)
        idx = rng.choice(X.shape[0], size=max_amostras, replace=False)
        X, rotulos = X[idx], rotulos[idx]

    lof_ = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contaminacao, n_jobs=-1,
    )
    lof_.fit_predict(X)
    score = lof_.negative_outlier_factor_

    out = pd.DataFrame({"rotulo": rotulos,
                        "score_anomalia_lof": score.astype("float32")})
    saida = config.caminho("outliers", "scores_lof.parquet")
    salvar_parquet(out, saida)
    print(f"[outliers] LOF rodado em {len(out):,} amostras")
    liberar(art, X, lof_)
    return saida


@com_gc
def executar(fazer_iforest=True, fazer_lof=False):
    saidas = {}
    if fazer_iforest:
        saidas["iforest"] = str(isolation_forest())
    if fazer_lof:
        saidas["lof"] = str(lof())
    return saidas
