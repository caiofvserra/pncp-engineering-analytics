"""Modelo online do sistema — o loop de aprendizado com o humano.

Escolha de projeto (barato + sem GPU + robusto):
  HashingVectorizer (stateless, sem vocabulário para persistir)
  + SGDClassifier(loss='log_loss') com partial_fit → aprendizado INCREMENTAL.

O SBERT/notebook produz a probabilidade base (prob_base). Este modelo aprende a
CORRIGIR essa base a partir do feedback humano; o score final é uma mistura das
duas. Antes de haver feedback suficiente, o score final = prob_base.

Robustez: o modelo vive num único arquivo joblib; o re-treino grava primeiro um
arquivo temporário e só então o substitui (troca atômica). Se algo falhar, o
modelo anterior permanece intacto.
"""
import os
import threading
import numpy as np
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from . import config

_LOCK = threading.Lock()
_VEC = HashingVectorizer(n_features=2 ** 18, alternate_sign=False,
                         ngram_range=(1, 2))
_clf = None            # SGDClassifier | None (None = ainda sem aprendizado)


def _carrega():
    global _clf
    if _clf is None and config.MODELO_ONLINE.exists():
        try:
            _clf = joblib.load(config.MODELO_ONLINE)
        except Exception:
            _clf = None
    return _clf


def treinado() -> bool:
    return _carrega() is not None and hasattr(_carrega(), "coef_")


def _limpa(texto: str) -> str:
    return " ".join(str(texto).lower().split())


def incrementa(textos: list[str], labels: list[int], peso: float = 3.0) -> int:
    """partial_fit incremental com o lote de feedback. Devolve nº de exemplos.
    Troca atômica do arquivo do modelo. Nunca lança para o chamador."""
    global _clf
    if not textos:
        return 0
    with _LOCK:
        try:
            X = _VEC.transform([_limpa(t) for t in textos])
            y = np.asarray(labels)
            w = np.full(len(y), float(peso))
            clf = _carrega()
            if clf is None:
                clf = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=42)
            # partial_fit exige conhecer todas as classes na 1ª chamada
            if not hasattr(clf, "coef_"):
                clf.partial_fit(X, y, classes=np.array([0, 1]), sample_weight=w)
            else:
                clf.partial_fit(X, y, sample_weight=w)
            tmp = str(config.MODELO_ONLINE) + ".tmp"
            joblib.dump(clf, tmp)
            os.replace(tmp, config.MODELO_ONLINE)   # atômico
            _clf = clf
            return len(y)
        except Exception as e:                       # nunca derruba o servidor
            return -1  # sinaliza falha; chamador loga


def prob_online(textos: list[str]):
    """Probabilidade do modelo online (ou None se ainda não treinado)."""
    if not treinado():
        return None
    try:
        X = _VEC.transform([_limpa(t) for t in textos])
        return _carrega().predict_proba(X)[:, 1]
    except Exception:
        return None


def score_final(textos: list[str], prob_base):
    """Mistura prob_base (notebook) com a correção do modelo online."""
    base = np.asarray([b if b is not None else 0.5 for b in prob_base], float)
    p = prob_online(textos)
    if p is None:
        return base
    a = float(config.CONFIG_PADRAO["blend_peso_online"])
    return np.clip((1 - a) * base + a * p, 0, 1)
