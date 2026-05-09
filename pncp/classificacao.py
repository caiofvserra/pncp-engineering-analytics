"""
Classificação supervisionada — baseline + rigor estatístico.

Modelos:
  - Regressão Logística (rápido, interpretável)
  - Random Forest (não-linear)
  - Linear SVC (margem máxima, bom em texto esparso)

Validação:
  - Holdout estratificado (test_size=0.2)
  - Cross-validation k=5
  - Teste de McNemar (LR vs RF)
  - Bootstrap dos F1 (intervalos de confiança)

Saídas em dados/classificacao/:
  - modelo_lr.joblib, modelo_rf.joblib, modelo_svc.joblib
  - metricas.json (F1 por classe, matriz de confusão, IC bootstrap)
  - ranking.parquet (contratos 'geral' ordenados por prob de engenharia)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    cross_val_score, train_test_split, GridSearchCV,
)
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
)

from pncp import config
from pncp.io_disco import (
    ler_parquet, salvar_parquet, salvar_modelo, salvar_json,
)
from pncp.ram import liberar, monitorar_ram, com_gc
from pncp.texto import carregar_tfidf


# ── Treino e métricas ────────────────────────────────────────────────────────
def _treinar_modelos(X_tr, y_tr, n_estimators_rf=100):
    modelos = {
        "lr": LogisticRegression(
            max_iter=1000, class_weight="balanced", n_jobs=-1,
            random_state=config.SEED, solver="saga",
        ),
        "rf": RandomForestClassifier(
            n_estimators=n_estimators_rf, class_weight="balanced", n_jobs=-1,
            random_state=config.SEED, max_depth=20,
        ),
        "svc": CalibratedClassifierCV(  # SVC linear + calibração para ter prob
            LinearSVC(class_weight="balanced", random_state=config.SEED),
            cv=2,
        ),
    }
    for nome, m in modelos.items():
        print(f"[clf] treinando {nome}...")
        m.fit(X_tr, y_tr)
    return modelos


def _avaliar(modelo, X_te, y_te):
    pred = modelo.predict(X_te)
    return {
        "f1_macro": float(f1_score(y_te, pred, average="macro")),
        "f1_engenharia": float(f1_score(y_te, pred, labels=["engenharia"],
                                        average="macro", zero_division=0)),
        "matriz_confusao": confusion_matrix(y_te, pred).tolist(),
        "relatorio": classification_report(y_te, pred, zero_division=0,
                                            output_dict=True),
    }


def _bootstrap_f1(modelo, X_te, y_te, n=None):
    """IC do F1 por reamostragem do conjunto de teste."""
    if n is None:
        n = config.N_BOOTSTRAP
    rng = np.random.default_rng(config.SEED)
    pred = modelo.predict(X_te)
    pred = np.asarray(pred)
    y_te = np.asarray(y_te)
    f1s = np.empty(n)
    idx = np.arange(len(y_te))
    for i in range(n):
        amostra = rng.choice(idx, size=len(idx), replace=True)
        f1s[i] = f1_score(y_te[amostra], pred[amostra],
                          labels=["engenharia"], average="macro",
                          zero_division=0)
    return {
        "f1_eng_media": float(f1s.mean()),
        "f1_eng_ic95": [float(np.percentile(f1s, 2.5)),
                         float(np.percentile(f1s, 97.5))],
    }


def _mcnemar(modelo_a, modelo_b, X_te, y_te):
    """Compara dois modelos no mesmo conjunto. Retorna p-valor."""
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        return {"erro": "instale statsmodels para teste de McNemar"}
    pa = modelo_a.predict(X_te) == y_te
    pb = modelo_b.predict(X_te) == y_te
    tabela = [
        [int(((pa) & (pb)).sum()), int(((pa) & (~pb)).sum())],
        [int(((~pa) & (pb)).sum()), int(((~pa) & (~pb)).sum())],
    ]
    res = mcnemar(tabela, exact=False, correction=True)
    return {"p_valor": float(res.pvalue), "estatistica": float(res.statistic),
            "tabela": tabela}


def _grid_search_lr(X_tr, y_tr):
    """GridSearch enxuto para LR (C). Reduzido para 2 valores × CV2 = 4 fits."""
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced",
                           n_jobs=-1, random_state=config.SEED, solver="saga"),
        param_grid={"C": [0.5, 2.0]},
        scoring="f1_macro",
        cv=2,
        n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


# ── Active learning: escolhe contratos mais informativos para revisão ──────
def amostra_active_learning(modelo, X, df_meta, n=50, estrategia="incerteza"):
    """
    Inspirado no Cap. 7.5.2 (Active Learning, Han/Kamber/Pei).

    Em vez de pegar os top-N pelo score (=mais óbvios), pega os N mais
    INFORMATIVOS para revisão humana. Estratégias:

      - 'incerteza': uncertainty sampling — predições com prob mais
        próxima de 0.5 (modelo está em dúvida).
      - 'margem': diferença pequena entre top-1 e top-2 classes.
      - 'entropia': entropia das probabilidades — alta entropia = incerto.

    Retorna DataFrame ordenado por incerteza (mais informativo primeiro).
    """
    if not hasattr(modelo, "predict_proba"):
        return pd.DataFrame()
    proba = modelo.predict_proba(X)

    if estrategia == "incerteza":
        # Para cada amostra, pega a prob máxima; quanto menor, mais incerto.
        score = -proba.max(axis=1)
    elif estrategia == "margem":
        ord_ = np.sort(proba, axis=1)
        score = -(ord_[:, -1] - ord_[:, -2])  # menor margem = mais incerto
    else:  # entropia
        score = -(-(proba * np.log(proba + 1e-12)).sum(axis=1))

    out = df_meta.copy()
    out["score_incerteza"] = score.astype("float32")
    # Foca em 'geral' (que é o cluster com risco de subenquadramento)
    if "rotulo" in out.columns:
        out = out[out["rotulo"] == "geral"]
    return out.sort_values("score_incerteza", ascending=False).head(n)


# ── Calibração de probabilidades ───────────────────────────────────────────
def calibrar(modelo, X_tr, y_tr, metodo="isotonic"):
    """
    Calibra as probabilidades do modelo (Platt sigmoid ou Isotonic).
    Importante quando vamos usar threshold (0.5 etc.) — sem calibração,
    o threshold pode estar deslocado.
    """
    from sklearn.calibration import CalibratedClassifierCV
    cal = CalibratedClassifierCV(modelo, method=metodo, cv="prefit")
    cal.fit(X_tr, y_tr)
    return cal


# ── Ranking de suspeitos ─────────────────────────────────────────────────────
def _gerar_ranking(modelo, X, df_meta):
    """Para os contratos rotulados 'geral', ordena pela prob de 'engenharia'."""
    if not hasattr(modelo, "predict_proba"):
        return pd.DataFrame()
    classes = list(modelo.classes_)
    if "engenharia" not in classes:
        return pd.DataFrame()
    idx_eng = classes.index("engenharia")
    proba = modelo.predict_proba(X)[:, idx_eng]
    out = df_meta.copy()
    out["prob_engenharia"] = proba.astype("float32")
    suspeitos = (out[out["rotulo"] == "geral"]
                 .sort_values("prob_engenharia", ascending=False))
    return suspeitos


# ── Pipeline ─────────────────────────────────────────────────────────────────
@com_gc
def executar(caminho_parquet=None,
             fazer_grid=True,
             fazer_holdout=True,
             fazer_mcnemar=True,
             fazer_bootstrap=True,
             fazer_cv=True,
             n_bootstrap=200,
             n_estimators_rf=100,
             cv_folds=3,
             max_amostras_treino=300_000,
             forcar=False):
    """
    Pipeline completo: TF-IDF (carregado do disco) → treino → avaliação →
    ranking → persistência.

    Args:
      forcar: se True, ignora resultado anterior e re-treina. Default False
              (skip se já existe metricas.json + ranking.parquet).
      n_bootstrap: nº de re-amostragens para IC. 200 dá IC bom; 1000 é mais
                   preciso mas 5× mais lento.
      n_estimators_rf: árvores no Random Forest. 100 é bom; 200 pouco ganho.
      cv_folds: 3 (rápido) vs 5 (mais preciso, mais lento).
      max_amostras_treino: se o dataset for maior que isso, usa subsample
                           ESTRATIFICADO para treino (preserva proporção
                           dos rótulos). None = usa tudo. Para 1M+ linhas,
                           subsample reduz tempo em ~3× sem perda significativa.
    """
    from pncp.ram import precisa_de
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")

    saida = config.caminho(config.SUB_P2)
    tfidf_X = saida / "X.npz"
    if not precisa_de(tfidf_X, "classificacao",
                       "rode pncp.texto.construir_tfidf(...) primeiro"):
        return None

    # Skip-if-exists: se métricas e ranking já estão salvos, pula
    metricas_path = saida / "metricas.json"
    ranking_path = saida / "ranking.parquet"
    if not forcar and metricas_path.exists() and ranking_path.exists():
        print(f"[clf] já rodou anteriormente — pulando "
              f"(use forcar=True para re-treinar)")
        return saida

    monitorar_ram("início clf")
    artefatos = carregar_tfidf()
    X = artefatos["X"]
    y = artefatos["labels"]["rotulo"].astype(str).values

    # Subsample estratificado para treino — RF e CV em 1M linhas é proibitivo
    if max_amostras_treino and X.shape[0] > max_amostras_treino:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1,
                                       train_size=max_amostras_treino,
                                       random_state=config.SEED)
        idx_sub, _ = next(sss.split(X, y))
        X_full, y_full = X, y
        X, y = X[idx_sub], y[idx_sub]
        print(f"[clf] subsample estratificado: {X_full.shape[0]:,} → "
              f"{X.shape[0]:,} (treino) — predição final usa todos")
    else:
        X_full, y_full = X, y

    metricas = {}

    if fazer_holdout:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y,
        )
    else:
        X_tr, X_te, y_tr, y_te = X, X, y, y

    if fazer_grid:
        print("[clf] grid search LR (2 Cs × CV2 = 4 fits)...")
        lr_best, params, cv_score = _grid_search_lr(X_tr, y_tr)
        metricas["grid_lr"] = {"melhores_params": params, "cv_f1_macro": cv_score}
    else:
        lr_best = None

    modelos = _treinar_modelos(X_tr, y_tr, n_estimators_rf=n_estimators_rf)
    if lr_best is not None:
        modelos["lr"] = lr_best

    monitorar_ram("após treino")

    metricas["holdout"] = {nome: _avaliar(m, X_te, y_te)
                           for nome, m in modelos.items()}

    if fazer_bootstrap:
        metricas["bootstrap"] = {
            nome: _bootstrap_f1(m, X_te, y_te, n=n_bootstrap)
            for nome, m in modelos.items()
        }

    if fazer_mcnemar:
        metricas["mcnemar_lr_vs_rf"] = _mcnemar(modelos["lr"], modelos["rf"],
                                                 X_te, y_te)

    if fazer_cv:
        metricas[f"cv{cv_folds}_f1_macro"] = {}
        for nome, m in modelos.items():
            scores = cross_val_score(m, X, y, cv=cv_folds,
                                       scoring="f1_macro", n_jobs=-1)
            metricas[f"cv{cv_folds}_f1_macro"][nome] = {
                "media": float(scores.mean()), "desvio": float(scores.std()),
            }

    # Ranking de suspeitos com o melhor modelo (por F1-engenharia)
    melhor = max(modelos, key=lambda n: metricas["holdout"][n]["f1_engenharia"])
    metricas["melhor_modelo"] = melhor
    # Para o ranking: usa o dataset COMPLETO (não o subsample) porque
    # queremos ranquear todos os 'geral', não só uma amostra
    df_meta = ler_parquet(caminho_parquet,
                          colunas=["numeroControlePNCP", "objeto", "rotulo",
                                   "anoPublicacao", "valor"])
    ranking = _gerar_ranking(modelos[melhor], X_full, df_meta)
    salvar_parquet(ranking.head(5000), saida / "ranking.parquet")

    # Amostra para revisão humana via active learning (uncertainty sampling)
    incertos = amostra_active_learning(modelos[melhor], X_full, df_meta, n=50)
    if not incertos.empty:
        salvar_parquet(incertos, saida / "amostra_active_learning.parquet")

    # Persiste modelos e métricas
    for nome, m in modelos.items():
        salvar_modelo(m, saida / f"modelo_{nome}.joblib")
    salvar_json(metricas, saida / "metricas.json")
    print(f"[clf] melhor={melhor} | F1-eng={metricas['holdout'][melhor]['f1_engenharia']:.4f}")

    liberar(X, modelos, df_meta, ranking)
    monitorar_ram("fim clf")
    mostrar()
    return saida


def mostrar():
    """Imprime resumo das métricas + matriz de confusão do melhor modelo."""
    from pncp.io_disco import ler_json
    p = config.caminho(config.SUB_P2, "metricas.json")
    if not p.exists():
        print("[clf.mostrar] rode pncp.classificacao.executar() primeiro")
        return
    m = ler_json(p)
    melhor = m.get("melhor_modelo", "?")
    print(f"\n📈 Classificação — melhor modelo: {melhor}")
    if "holdout" in m and melhor in m["holdout"]:
        h = m["holdout"][melhor]
        print(f"   F1-engenharia (holdout): {h.get('f1_engenharia', 0):.4f}")
        print(f"   F1-macro:                {h.get('f1_macro', 0):.4f}")
    if "bootstrap" in m and melhor in m["bootstrap"]:
        ic = m["bootstrap"][melhor].get("f1_eng_ic95")
        if ic:
            print(f"   IC 95%: [{ic[0]:.3f}, {ic[1]:.3f}]")
    if "mcnemar_lr_vs_rf" in m:
        print(f"   McNemar LR vs RF: p={m['mcnemar_lr_vs_rf'].get('p_valor', 0):.4f}")
    rk = config.caminho(config.SUB_P2, "ranking.parquet")
    if rk.exists():
        from pncp.io_disco import ler_parquet
        r = ler_parquet(rk).head(10)
        print(f"\n   Top-10 suspeitos (rotulo='geral' com maior prob_engenharia):")
        for _, row in r.iterrows():
            txt = str(row.get("objeto", ""))[:80]
            prob = row.get("prob_engenharia", 0)
            print(f"   • [{prob:.3f}] {txt}")
