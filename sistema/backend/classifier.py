"""Classificador PRÓPRIO do sistema (não depende do notebook).

Aprendizado fraco supervisionado + humano no loop, barato e sem GPU:
  - positivos: contratos que o órgão já rotula como engenharia/obras (confiáveis);
  - negativos: amostra de "serviços gerais" (majoritariamente não-engenharia);
  - + rótulos da TRIAGEM humana, com peso maior (corrigem os casos difíceis).
Representação TF-IDF (uni+bi-gramas) + Regressão Logística calibrada. O
re-treino é um FIT completo (segundos sobre dezenas de milhares de textos) —
mais simples e estável que treino online, e roda no CPU.

Robusto: treino grava em arquivo temporário e troca atômica; se falhar, mantém
o modelo anterior. Sem modelo treinado, score() devolve 0.5 (neutro)."""
import os
import threading
import re
import unicodedata
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from . import config, db

_LOCK = threading.Lock()
_MODELO = None          # (vectorizer, clf) | None


def _norm(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", t)


def _carrega():
    global _MODELO
    if _MODELO is None and config.MODELO.exists():
        try:
            _MODELO = joblib.load(config.MODELO)
        except Exception:
            _MODELO = None
    return _MODELO


def treinado():
    return _carrega() is not None


def treinar():
    """(Re)treina do zero com o corpus atual + rótulos humanos. Devolve métricas."""
    with _LOCK:
        corpus = db.corpus_treino()          # [(objeto, y, peso)]
        if len(corpus) < 12 or len({y for _, y, _ in corpus}) < 2:
            db.evento("modelo", "treino adiado: corpus insuficiente")
            return {"ok": False, "msg": "Corpus insuficiente para treinar."}
        try:
            textos = [_norm(o) for o, _, _ in corpus]
            y = np.array([yy for _, yy, _ in corpus])
            w = np.array([ww for _, _, ww in corpus], dtype=float)
            _mdf = 1 if len(corpus) < 500 else 2   # bases pequenas: aceita termos raros
            vec = TfidfVectorizer(min_df=_mdf, ngram_range=(1, 2), max_features=40000)
            X = vec.fit_transform(textos)
            base = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
            base.fit(X, y, sample_weight=w)
            # calibração: probabilidades honestas para o limiar fazer sentido
            try:
                clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
                clf.fit(X, y, sample_weight=w)
            except Exception:
                clf = base
            tmp = str(config.MODELO) + ".tmp"
            joblib.dump((vec, clf), tmp)
            os.replace(tmp, config.MODELO)
            global _MODELO
            _MODELO = (vec, clf)
            met = {"ok": True, "n_treino": len(corpus),
                   "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
                   "n_humano": int((w > 1).sum())}
            db.evento("modelo", f"Treinado: {met['n_pos']} pos / {met['n_neg']} neg "
                      f"(+{met['n_humano']} rótulos humanos).")
            return met
        except Exception as e:
            db.evento("modelo", f"Falha no treino: {str(e)[:120]}")
            return {"ok": False, "msg": f"Falha no treino: {e}"}


def score(textos):
    """Probabilidade de ser engenharia para cada texto (0.5 se sem modelo)."""
    m = _carrega()
    if m is None:
        return np.full(len(textos), 0.5)
    vec, clf = m
    try:
        return clf.predict_proba(vec.transform([_norm(t) for t in textos]))[:, 1]
    except Exception:
        return np.full(len(textos), 0.5)
