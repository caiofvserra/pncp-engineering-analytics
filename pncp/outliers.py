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
def one_class_svm(nu=0.05, kernel="rbf"):
    """
    One-Class SVM (Cap. 11.5.2). Treina apenas em "normais" do cluster
    'geral' e marca como outlier qualquer ponto fora da fronteira.
    Mais sensível que IsolationForest em texto-derivado.
    """
    from sklearn.svm import OneClassSVM

    art = carregar_tfidf()
    X = _reduzir_dim(art["X"], n=50)
    rotulos = art["labels"]["rotulo"].astype(str).values
    mask_geral = rotulos == "geral"
    if mask_geral.sum() < 100:
        return None

    # Subsample (OCSVM é O(n²))
    rng = np.random.default_rng(config.SEED)
    treino = rng.choice(np.where(mask_geral)[0],
                          size=min(5000, mask_geral.sum()), replace=False)
    svm = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")
    svm.fit(X[treino])
    score = svm.score_samples(X)

    out = pd.DataFrame({"rotulo": rotulos,
                        "score_anomalia_ocsvm": score.astype("float32")})
    saida = config.caminho("outliers", "scores_ocsvm.parquet")
    salvar_parquet(out, saida)
    print(f"[outliers] One-Class SVM rodado")
    liberar(art, X, svm)
    return saida


def zscore_valor(caminho_parquet=None, limiar_z=3.0):
    """
    Detecção univariada (Cap. 11.2.1, Han/Kamber/Pei).
    Usa Z-score no `valor` por categoria — contratos rotulados 'geral'
    com valor anormalmente alto dentro do próprio rótulo são flag.
    Também aplica regra do IQR (>Q3+1.5×IQR).
    """
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")
    df = ler_parquet(caminho_parquet,
                     colunas=["numeroControlePNCP", "rotulo", "valor"])
    if "valor" not in df.columns:
        print("[outliers] coluna 'valor' ausente — pulando z-score")
        return None

    g = df[df["rotulo"] == "geral"].copy()
    g["valor"] = pd.to_numeric(g["valor"], errors="coerce")
    g = g.dropna(subset=["valor"])

    mu, sigma = g["valor"].mean(), g["valor"].std()
    g["zscore_valor"] = (g["valor"] - mu) / max(sigma, 1e-9)

    q1, q3 = g["valor"].quantile([0.25, 0.75])
    iqr = q3 - q1
    g["acima_iqr"] = g["valor"] > (q3 + 1.5 * iqr)
    g["outlier_z"] = g["zscore_valor"].abs() >= limiar_z

    saida = config.caminho("outliers", "valor_outliers.parquet")
    salvar_parquet(g[g["outlier_z"] | g["acima_iqr"]], saida)
    salvar_json({
        "media": float(mu), "desvio": float(sigma),
        "Q1": float(q1), "Q3": float(q3), "IQR": float(iqr),
        "n_outlier_z": int(g["outlier_z"].sum()),
        "n_acima_iqr": int(g["acima_iqr"].sum()),
    }, config.caminho("outliers", "resumo_valor.json"))
    print(f"[outliers] valor: {int(g['outlier_z'].sum())} z-outliers, "
          f"{int(g['acima_iqr'].sum())} acima do IQR")
    liberar(df, g)
    return saida


@com_gc
def ensemble():
    """
    Ensemble de detectores (Cap. 11.7.3): combina IsolationForest + LOF +
    OCSVM via min-max normalization + média. Um contrato é "outlier
    consenso" se aparece nos top-K de pelo menos 2 detectores.
    """
    paths = {
        "iforest": config.caminho("outliers", "scores_iforest.parquet"),
        "lof": config.caminho("outliers", "scores_lof.parquet"),
        "ocsvm": config.caminho("outliers", "scores_ocsvm.parquet"),
    }
    tabelas = {}
    for nome, p in paths.items():
        if Path(p).exists():
            tabelas[nome] = ler_parquet(p)
    if len(tabelas) < 2:
        print("[outliers] ensemble precisa de ≥2 detectores rodados")
        return None

    # Min-max normalize cada score (quanto menor o original, mais outlier;
    # convertemos para 'maior = mais outlier' para a média).
    def norm(serie):
        s = -serie  # inverte
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s

    base = None
    for nome, tab in tabelas.items():
        col = [c for c in tab.columns if c.startswith("score_anomalia_")][0]
        tab = tab[["rotulo", col]].rename(columns={col: nome}).reset_index(drop=True)
        tab[nome] = norm(tab[nome])
        base = tab if base is None else pd.concat([base, tab[[nome]]], axis=1)

    cols_score = [c for c in tabelas.keys() if c in base.columns]
    base["score_ensemble_media"] = base[cols_score].mean(axis=1)
    base["score_ensemble_max"] = base[cols_score].max(axis=1)
    saida = config.caminho("outliers", "ensemble.parquet")
    salvar_parquet(base, saida)
    print(f"[outliers] ensemble com {len(cols_score)} detectores")
    return saida


# ── Imports tardios para Path ────────────────────────────────────────────────
from pathlib import Path  # noqa: E402


@com_gc
def executar(fazer_iforest=True, fazer_lof=False, fazer_ocsvm=True,
             fazer_zscore=True, fazer_ensemble=True, forcar=False):
    from pncp.ram import precisa_de
    if not precisa_de(config.caminho(config.SUB_P2, "X.npz"), "outliers",
                       "rode pncp.texto.construir_tfidf(...) primeiro"):
        return None
    # Skip inteligente: pula se outliers SÃO MAIS NOVOS que TF-IDF.
    from pncp.ram import cache_valido
    saida = config.caminho("outliers")
    tfidf = config.caminho(config.SUB_P2, "X.npz")
    if not forcar and cache_valido(saida / "scores_iforest.parquet", tfidf):
        print(f"[outliers] já rodou e está atualizado — pulando")
        return None
    if not forcar and (saida / "scores_iforest.parquet").exists():
        print(f"[outliers] TF-IDF é mais novo — refazendo")
    saidas = {}
    if fazer_iforest:
        saidas["iforest"] = str(isolation_forest())
    if fazer_lof:
        saidas["lof"] = str(lof())
    if fazer_ocsvm:
        saidas["ocsvm"] = str(one_class_svm())
    if fazer_zscore:
        saidas["zscore_valor"] = str(zscore_valor())
    if fazer_ensemble:
        ens = ensemble()
        if ens:
            saidas["ensemble"] = str(ens)
    return saidas
