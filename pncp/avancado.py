"""
Técnicas avançadas — não-supervisionado e semi-supervisionado.

Cada função é OPT-IN e independente:
  - LDA (tópicos latentes)
  - Label Propagation (semi-supervisionado: usa rótulos parciais)
  - Apriori (regras de associação em metadados)
  - KMeans + Silhouette (clustering)
  - Hierárquico (dendrograma de top suspeitos)
  - SMOTE (balanceamento, opcional pré-treino)

Cada uma lê TF-IDF do disco, salva resultados em dados/avancado/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pncp import config
from pncp.io_disco import (
    ler_parquet, salvar_parquet, salvar_json, salvar_modelo,
)
from pncp.ram import liberar, com_gc, monitorar_ram
from pncp.texto import carregar_tfidf


# ── LDA: tópicos latentes ────────────────────────────────────────────────────
def lda(n_topicos=8, n_palavras=10):
    """Latent Dirichlet Allocation sobre TF-IDF. Salva tópicos em JSON."""
    from sklearn.decomposition import LatentDirichletAllocation
    art = carregar_tfidf()
    X, vec = art["X"], art["vec"]
    modelo = LatentDirichletAllocation(
        n_components=n_topicos, random_state=config.SEED, n_jobs=-1,
        learning_method="online", batch_size=2048,
    )
    modelo.fit(X)
    vocab = vec.get_feature_names_out()

    topicos = []
    for i, comp in enumerate(modelo.components_):
        top = comp.argsort()[: -n_palavras - 1: -1]
        topicos.append({
            "id": i,
            "palavras": [vocab[j] for j in top],
            "pesos": [float(comp[j]) for j in top],
        })

    saida = config.caminho(config.SUB_P3, "lda_topicos.json")
    salvar_json({"n_topicos": n_topicos, "topicos": topicos}, saida)
    salvar_modelo(modelo, config.caminho(config.SUB_P3, "lda_modelo.joblib"))
    print(f"[avancado] LDA → {saida}")
    liberar(X, modelo)
    return saida


# ── Label Propagation: semi-supervisionado ───────────────────────────────────
def label_propagation(frac_rotulada=0.3, max_amostras=20000):
    """
    Esconde parte dos rótulos e propaga via grafo de similaridade.
    Útil quando suspeitamos que muitos 'geral' são na verdade 'engenharia'.
    """
    from sklearn.semi_supervised import LabelPropagation
    from sklearn.decomposition import TruncatedSVD

    art = carregar_tfidf()
    X = art["X"]
    y = art["labels"]["rotulo"].astype(str).values

    # Reduz dimensionalidade — Label Propagation não escala em alta dim.
    if X.shape[0] > max_amostras:
        idx = np.random.default_rng(config.SEED).choice(
            X.shape[0], size=max_amostras, replace=False,
        )
        X = X[idx]
        y = y[idx]

    svd = TruncatedSVD(n_components=64, random_state=config.SEED)
    X_red = svd.fit_transform(X).astype("float32")

    rng = np.random.default_rng(config.SEED)
    mask_oculta = rng.random(len(y)) > frac_rotulada
    y_treino = y.copy()
    y_treino[mask_oculta] = "-1"

    modelo = LabelPropagation(kernel="knn", n_neighbors=10, n_jobs=-1)
    modelo.fit(X_red, y_treino)
    pred = modelo.transduction_

    metricas = {
        "n_amostras": int(len(y)),
        "n_ocultas": int(mask_oculta.sum()),
        "acuracia_recuperacao": float((pred[mask_oculta] == y[mask_oculta]).mean()),
        "distrib_predicao": pd.Series(pred).value_counts().to_dict(),
    }
    saida = config.caminho(config.SUB_P3, "label_propagation.json")
    salvar_json(metricas, saida)
    print(f"[avancado] Label Propagation → recuperação="
          f"{metricas['acuracia_recuperacao']:.3f}")
    liberar(X, X_red, modelo)
    return saida


# ── Apriori: regras de associação em metadados ──────────────────────────────
def apriori(min_support=0.05, min_confidence=0.6):
    """
    Encontra regras tipo 'modalidade=Pregão & valor>X → rotulo=geral'.
    Trabalha apenas com colunas categóricas/binadas, não TF-IDF.
    """
    try:
        from mlxtend.frequent_patterns import apriori as _ap, association_rules
    except ImportError:
        print("[avancado] mlxtend não instalado — pip install mlxtend")
        return None

    caminho = config.caminho(config.SUB_COLETA, "contratos.parquet")
    df = ler_parquet(caminho)

    # Discretiza para boolean
    cols_cat = [c for c in ("modalidadeNome", "rotulo") if c in df.columns]
    bin_df = pd.get_dummies(df[cols_cat].astype(str), prefix_sep="=")
    if "valor" in df.columns:
        bin_df["valor_alto"] = df["valor"] > df["valor"].quantile(0.75)
        bin_df["valor_baixo"] = df["valor"] < df["valor"].quantile(0.25)

    itens = _ap(bin_df, min_support=min_support, use_colnames=True)
    regras = association_rules(itens, metric="confidence",
                                min_threshold=min_confidence)
    regras = regras.sort_values("lift", ascending=False).head(50)
    # Converte frozensets para lista de strings (parquet não aceita frozenset)
    regras["antecedents"] = regras["antecedents"].apply(lambda s: list(s))
    regras["consequents"] = regras["consequents"].apply(lambda s: list(s))

    saida = config.caminho(config.SUB_P3, "apriori_regras.parquet")
    salvar_parquet(regras, saida)
    print(f"[avancado] Apriori → {len(regras)} regras em {saida}")
    liberar(df, bin_df, itens, regras)
    return saida


# ── KMeans + Silhouette ─────────────────────────────────────────────────────
def kmeans(k_max=10):
    """Roda KMeans para k=2..k_max e salva silhouette por k."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import TruncatedSVD

    art = carregar_tfidf()
    svd = TruncatedSVD(n_components=50, random_state=config.SEED)
    X = svd.fit_transform(art["X"]).astype("float32")

    # Subsample para silhouette (caro em N grande)
    rng = np.random.default_rng(config.SEED)
    sub = rng.choice(X.shape[0], size=min(5000, X.shape[0]), replace=False)

    metricas = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=config.SEED)
        rotulos = km.fit_predict(X)
        sil = silhouette_score(X[sub], rotulos[sub])
        metricas.append({"k": k, "inercia": float(km.inertia_),
                         "silhouette": float(sil)})

    saida = config.caminho(config.SUB_P3, "kmeans.json")
    salvar_json(metricas, saida)
    print(f"[avancado] KMeans → {saida}")
    liberar(X, art)
    return saida


# ── Clustering hierárquico nos top suspeitos ────────────────────────────────
def hierarquico_suspeitos(n_amostra=30):
    """Dendrograma dos top-N suspeitos para inspeção visual."""
    from scipy.cluster.hierarchy import linkage, dendrogram

    ranking_path = config.caminho(config.SUB_P2, "ranking.parquet")
    if not ranking_path.exists():
        print("[avancado] rode classificacao.executar() antes")
        return None

    ranking = ler_parquet(ranking_path).head(n_amostra)
    art = carregar_tfidf()
    X = art["X"]
    # Recupera linhas pelo índice — assume mesma ordem de carregar_tfidf
    # (se ranking guarda indice original, melhor; aqui usamos as primeiras)
    idx = ranking.index.to_numpy()
    idx = idx[idx < X.shape[0]]
    sub = X[idx].toarray()

    Z = linkage(sub, method="ward")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, ax=ax, labels=[f"#{i}" for i in idx])
    ax.set_title("Hierárquico — top suspeitos")
    saida = config.caminho(config.SUB_P3, "hierarquico.png")
    fig.tight_layout()
    fig.savefig(saida, dpi=120, bbox_inches="tight")
    plt.close(fig)
    liberar(art, sub, Z)
    return saida


# ── SMOTE — gera amostras sintéticas da classe minoritária ──────────────────
def smote_balancear():
    """Aplica SMOTE no TF-IDF e salva matriz balanceada."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("[avancado] imblearn não instalado")
        return None
    from pncp.io_disco import salvar_sparse

    art = carregar_tfidf()
    X, y = art["X"], art["labels"]["rotulo"].astype(str).values
    sm = SMOTE(random_state=config.SEED, n_jobs=-1)
    Xb, yb = sm.fit_resample(X, y)
    saida = config.caminho(config.SUB_P3)
    salvar_sparse(Xb, saida / "X_smote.npz")
    salvar_parquet(pd.DataFrame({"rotulo": yb}), saida / "y_smote.parquet")
    print(f"[avancado] SMOTE: {X.shape[0]:,} → {Xb.shape[0]:,}")
    liberar(art, X, Xb)
    return saida


# ── GMM / EM: soft clustering ────────────────────────────────────────────────
def gmm(n_componentes=3):
    """
    Gaussian Mixture Model treinado via EM (Cap. 9.1.3, Han/Kamber/Pei).
    Diferente do KMeans: cada contrato recebe **probabilidade** de
    pertencer a cada cluster — permite identificar contratos "no meio
    do caminho" entre 'geral' e 'engenharia'.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import TruncatedSVD

    art = carregar_tfidf()
    X = TruncatedSVD(n_components=50, random_state=config.SEED).fit_transform(art["X"])
    X = X.astype("float32")

    gm = GaussianMixture(n_components=n_componentes, random_state=config.SEED,
                          covariance_type="diag", max_iter=200)
    gm.fit(X)
    proba = gm.predict_proba(X).astype("float32")

    # Salva probas + rótulo original; permite cruzamento downstream
    rotulos = art["labels"]["rotulo"].astype(str).values
    out = pd.DataFrame(proba, columns=[f"prob_cluster_{i}" for i in range(n_componentes)])
    out["rotulo"] = rotulos
    out["entropia"] = -(proba * np.log(proba + 1e-12)).sum(axis=1).astype("float32")

    saida = config.caminho(config.SUB_P3, "gmm.parquet")
    salvar_parquet(out, saida)
    print(f"[avancado] GMM/EM com {n_componentes} componentes → {saida}")
    liberar(X, gm, art)
    return saida


# ── Pipeline completo (chama tudo ligado por padrão) ─────────────────────────
@com_gc
def executar(fazer_lda=True, fazer_lp=True, fazer_apriori=True,
             fazer_kmeans=True, fazer_hier=True, fazer_gmm=True,
             fazer_smote=False):
    """Roda todas as técnicas avançadas (cada uma é opt-in)."""
    monitorar_ram("início avancado")
    saidas = {}
    if fazer_lda:
        saidas["lda"] = str(lda())
    if fazer_lp:
        saidas["label_propagation"] = str(label_propagation())
    if fazer_apriori:
        saidas["apriori"] = str(apriori())
    if fazer_kmeans:
        saidas["kmeans"] = str(kmeans())
    if fazer_hier:
        saidas["hierarquico"] = str(hierarquico_suspeitos())
    if fazer_gmm:
        saidas["gmm"] = str(gmm())
    if fazer_smote:
        saidas["smote"] = str(smote_balancear())
    salvar_json(saidas, config.caminho(config.SUB_P3, "indice.json"))
    monitorar_ram("fim avancado")
    return saidas
