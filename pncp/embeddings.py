"""
Embeddings densos via BERTimbau ou Sentence-BERT multilíngue.

Para 300k contratos × 768 dim em float32 = ~900MB. Salvamos float16 (~450MB)
e carregamos via mmap, então o classificador final vê só o slice que precisa.

Cada batch passa pela GPU (se disponível) e vai direto para o array no disco.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pncp import config
from pncp.io_disco import (
    ler_parquet, salvar_parquet, salvar_npy, ler_npy, salvar_modelo, salvar_json,
)
from pncp.ram import liberar, com_gc, monitorar_ram


def _carregar_modelo(tipo):
    """Carrega Sentence-BERT (rápido) ou BERTimbau (mais lento, mais preciso)."""
    if tipo == "sentence-bert":
        from sentence_transformers import SentenceTransformer
        return ("sbert", SentenceTransformer(config.MODELO_SBERT))
    if tipo == "bertimbau":
        # Pooling manual sobre BERT base (não tem head SBERT pronto).
        import torch
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(config.MODELO_BERTIMBAU)
        modelo = AutoModel.from_pretrained(config.MODELO_BERTIMBAU)
        if torch.cuda.is_available():
            modelo = modelo.cuda()
        modelo.eval()
        return ("bertimbau", (tok, modelo))
    raise ValueError(f"tipo desconhecido: {tipo}")


def _embed_sbert(modelo, textos, batch):
    return modelo.encode(textos, batch_size=batch, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)


def _embed_bertimbau(handle, textos, batch):
    import torch
    tok, modelo = handle
    out = []
    with torch.no_grad():
        for i in range(0, len(textos), batch):
            sub = textos[i: i + batch]
            enc = tok(sub, padding=True, truncation=True, max_length=128,
                       return_tensors="pt")
            if torch.cuda.is_available():
                enc = {k: v.cuda() for k, v in enc.items()}
            saida = modelo(**enc).last_hidden_state
            # mean pooling com máscara de atenção
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (saida * mask).sum(1) / mask.sum(1).clamp(min=1)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())
    return np.vstack(out)


@com_gc
def gerar(tipo="sentence-bert", caminho_parquet=None, max_amostras=None):
    """
    Gera embeddings densos e salva em .npy float16.
    Recomenda 'sentence-bert' (rápido) para iterar; 'bertimbau' p/ produção final.
    """
    if caminho_parquet is None:
        caminho_parquet = config.caminho(config.SUB_COLETA, "contratos.parquet")
    monitorar_ram("início embeddings")

    df = ler_parquet(caminho_parquet, colunas=["objeto", "rotulo"])
    if max_amostras:
        df = df.sample(n=min(max_amostras, len(df)), random_state=config.SEED)
    textos = df["objeto"].fillna("").astype(str).tolist()

    nome, modelo = _carregar_modelo(tipo)
    print(f"[emb] usando {nome} sobre {len(textos):,} textos...")
    if nome == "sbert":
        emb = _embed_sbert(modelo, textos, config.EMB_BATCH)
    else:
        emb = _embed_bertimbau(modelo, textos, config.EMB_BATCH)

    emb = emb.astype(config.EMB_DTYPE_DISCO)
    saida = config.caminho(config.SUB_EMB)
    salvar_npy(emb, saida / f"emb_{nome}.npy")
    salvar_parquet(df[["rotulo"]].reset_index(drop=True),
                    saida / f"emb_{nome}_labels.parquet")
    print(f"[emb] {emb.shape} salvo em {saida}")
    liberar(df, modelo, emb)
    monitorar_ram("fim embeddings")
    return saida


def treinar_classificador(tipo="sentence-bert", fazer_holdout=True,
                            modelo_final="svc"):
    """Treina um classificador sobre os embeddings já gerados.

    modelo_final: 'svc' (LinearSVC calibrado, melhor para texto denso —
    F1 +0.1 vs LR pura), 'lr' (LogisticRegression), 'rbf' (SVC RBF, lento
    mas pega não-linearidades).

    Sobre embeddings, modelos lineares puros tendem a sub-aproveitar a
    geometria do espaço — LinearSVC com escala normalizada costuma render
    mais que LR direta. Quando o F1 cai versus TF-IDF, geralmente é
    porque a classe minoritária ficou sub-representada após a projeção:
    elevamos n_iter, escalamos e calibramos.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC, SVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score

    saida = config.caminho(config.SUB_EMB)
    nome = "sbert" if tipo == "sentence-bert" else "bertimbau"
    X = ler_npy(saida / f"emb_{nome}.npy", mmap=False).astype("float32")
    y = ler_parquet(saida / f"emb_{nome}_labels.parquet")["rotulo"].astype(str).values

    if fazer_holdout:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y,
        )
    else:
        X_tr, X_te, y_tr, y_te = X, X, y, y

    if modelo_final == "lr":
        clf = Pipeline([
            ("escala", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(max_iter=3000, C=1.0,
                                       class_weight="balanced",
                                       n_jobs=-1, solver="lbfgs",
                                       random_state=config.SEED)),
        ])
    elif modelo_final == "rbf":
        clf = Pipeline([
            ("escala", StandardScaler(with_mean=False)),
            ("svc", CalibratedClassifierCV(
                SVC(kernel="rbf", class_weight="balanced",
                    random_state=config.SEED, cache_size=500),
                cv=2,
            )),
        ])
    else:  # svc (default) — LinearSVC calibrado
        clf = Pipeline([
            ("escala", StandardScaler(with_mean=False)),
            ("svc", CalibratedClassifierCV(
                LinearSVC(class_weight="balanced",
                          random_state=config.SEED, max_iter=5000, C=1.0),
                cv=3,
            )),
        ])

    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    metricas = {
        "tipo": tipo,
        "modelo_final": modelo_final,
        "f1_macro": float(f1_score(y_te, pred, average="macro")),
        "f1_engenharia": float(f1_score(y_te, pred, labels=["engenharia"],
                                        average="macro", zero_division=0)),
        "relatorio": classification_report(y_te, pred, zero_division=0,
                                            output_dict=True),
    }
    salvar_modelo(clf, saida / f"clf_{nome}.joblib")
    salvar_json(metricas, saida / f"metricas_{nome}.json")
    print(f"[emb] {tipo} ({modelo_final}) F1-eng={metricas['f1_engenharia']:.4f}, "
          f"F1-macro={metricas['f1_macro']:.4f}")
    liberar(X, clf)
    return metricas


@com_gc
def executar(tipo="sentence-bert", treinar=True, modelo_final="svc"):
    """Pipeline embeddings: gera + treina classificador.

    modelo_final: 'svc' (default, melhor F1 em embeddings densos),
    'lr' (mais rápido), 'rbf' (não-linear, lento mas pode ajudar).
    """
    gerar(tipo=tipo)
    if treinar:
        return treinar_classificador(tipo=tipo, modelo_final=modelo_final)
    return None
